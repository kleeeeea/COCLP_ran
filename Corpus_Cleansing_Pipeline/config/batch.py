# data_pipeline_refactor.py
"""
通用并发处理框架（**与业务解耦**）
================================

> **核心思想**：把 *线程池并发*、*模型端点轮询*、*批量文件遍历*、*断点续跑* 等**横切关注点**完全抽离出来，业务层只需实现一个 *record → record* 的纯函数即可。
>
> 这样无论你要做 *分类*、*纠错*、*摘要* 还是任何别的任务，只要把对应的处理函数塞进来，就能复用全部“基础设施”。

变更概要
--------
* 原来的 `ClassifyPipeline`、`classifier_fn` 等**分类特定命名**已改为通用：
  * `TaskPipeline` – 负责并发驱动与落盘。
  * `process_fn` – 你的业务函数签名 `def fn(record, dispatcher) -> record:`。
  * `batch_process()` – 针对整个文件夹的批量入口。
* 其余接口保持向后兼容；旧名字仍可用（`ClassifyPipeline = TaskPipeline`，见底部）。

用法示例
--------
```python
from data_pipeline_refactor import batch_process, ModelDispatcher

def my_new_task(rec, dsp: ModelDispatcher):
    # ... 在这里写你的逻辑 ...
    return rec  # 返回增强/修改后的 dict

batch_process(
    input_folder="./input",
    output_folder="./output",
    api_conns=[{"api_key": "None", "base_url": "https://example.com/v1"}],
    process_fn=my_new_task,
    max_workers=64,
)
```
"""
from __future__ import annotations
import time
import itertools
import json
import logging
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

# ----------------------------- 日志配置 --------------------------------------
LOG_FMT = "[%(asctime)s] %(levelname)s: %(message)s"
logging.basicConfig(format=LOG_FMT, datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------- 通用重试装饰器 ----------------------------------
from functools import wraps
from time import sleep

def retry(times: int = 3, backoff: float = 1.5):
    """可配置的指数退避重试装饰器。"""

    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            delay = 1.0
            for i in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:  # noqa: BLE001 – 捕获所有异常记录
                    if i == times - 1:
                        raise
                    logger.warning("%s 失败 (%s)，%.1fs 后重试…", fn.__name__, e, delay)
                    sleep(delay)
                    delay *= backoff
        return wrapper

    return deco

# --------------------------- 模型客户端分发器 ---------------------------------
class ModelDispatcher:
    def __init__(self, api_conns: Sequence[Dict[str, str]]):
        from openai import OpenAI
        self._OpenAI = OpenAI
        self._api_conns = list(api_conns)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._round_robin = itertools.cycle(self._api_conns)

    def _pick(self):
        with self._lock:
            return next(self._round_robin)

    def _get_client(self):
        if not hasattr(self._local, "client"):
            cfg = self._pick()
            self._local.client = self._OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
        return self._local.client

    @retry(times=4)
    def chat(self, *args, **kwargs):
        return self._get_client().chat.completions.create(*args, **kwargs)


# -------------------------- 线程池执行器 -------------------------------------
class ThreadedExecutor:
    """极简线程池：并发执行任务并 yield 结果。"""

    def __init__(self, max_workers: int = 32):
        self.max_workers = max_workers

    def run(self, fn: Callable[[Any], Any], items: Iterable[Any]):
        with ThreadPoolExecutor(self.max_workers) as pool:
            futures = {pool.submit(fn, it): it for it in items}
            for fut in as_completed(futures):
                yield fut.result()
class SmartAPIDispatcher:
    def __init__(self, api_conns: Sequence[Dict[str, str]]):
        from openai import OpenAI
        self._OpenAI = OpenAI
        self.api_conns = list(api_conns)
        
        # 状态跟踪
        self.usage_count = [0] * len(api_conns)
        self.error_count = [0] * len(api_conns)
        self.last_used = [0] * len(api_conns)  # 时间戳
        self.lock = threading.Lock()
        
        # 客户端缓存
        self.clients = [self._create_client(i) for i in range(len(api_conns))]
    
    def _create_client(self, index):
        cfg = self.api_conns[index]
        return self._OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    
    def get_client(self):
        """智能选择最优API客户端"""
        with self.lock:
            # 计算可用性分数（越高越好）
            scores = []
            current_time = time.time()
            
            for i in range(len(self.api_conns)):
                # 基础分 = 100 - 错误计数*10
                score = 100 - self.error_count[i] * 10
                
                # 增加空闲奖励（30分钟内未使用+50分）
                if current_time - self.last_used[i] > 30:
                    score += 50
                
                # 增加低使用率奖励
                usage_ratio = self.usage_count[i] / (sum(self.usage_count) or 1)
                score += int((1 - usage_ratio) * 30)
                
                scores.append(score)
            
            # 选择最高分的API
            selected = scores.index(max(scores))
            self.usage_count[selected] += 1
            self.last_used[selected] = current_time
            
            return self.clients[selected], selected
    
    @retry(times=3)
    def chat(self, *args, **kwargs):
        client, idx = self.get_client()
        try:
            start = time.time()
            response = client.chat.completions.create(*args, **kwargs)
            latency = time.time() - start
            
            # 成功时减少错误计数
            with self.lock:
                if self.error_count[idx] > 0:
                    self.error_count[idx] -= 1
                    
            logger.debug(f"API {idx} 成功 | 延迟: {latency:.2f}s")
            return response
        
        except Exception as e:
            with self.lock:
                self.error_count[idx] += 1
            logger.warning(f"API {idx} 错误: {str(e)}")
            raise
# --------------------------- JSONL 断点写入器 -------------------------------
class JsonlCheckpointer:
    """线程安全持久化 + checkpoint。"""

    def __init__(self, output_path: os.PathLike, id_field: str = "id", state_file: os.PathLike | None = None, record_dir:str="./record"):
        self.output_path = Path(output_path)
        self.id_field = id_field

        if state_file is None:
            record_dir = record_dir/"record"
            record_dir.mkdir(parents=True, exist_ok=True)  # 确保文件夹存在
            self.state_file = record_dir / f"{self.output_path.stem}_done.txt"
        else:
            self.state_file = Path(state_file)

        self._lock = threading.Lock()
        self.processed: set[str] = self._load_state()

    def _load_state(self) -> set[str]:
        if not self.state_file.exists():
            return set()
        with self.state_file.open("r", encoding="utf-8") as fh:
            return {line.strip() for line in fh if line.strip()}

    def _persist_state(self, _id: str):
        with self.state_file.open("a", encoding="utf-8") as fh:
            fh.write(f"{_id}\n")

    # -------------------------- API ------------------------------------
    def already_done(self, _id: str | None) -> bool:
        return _id in self.processed if _id else True

    def write(self, record: Dict[str, Any]):
        _id = record.get(self.id_field)
        with self._lock:
            if _id and _id not in self.processed:
                with self.output_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.processed.add(_id)
                self._persist_state(_id)

# ----------------------------- 主流水线 --------------------------------------
class TaskPipeline:
    """并发调度 + 断点续跑的通用流水线。"""

    Record = Dict[str, Any]
    ProcessFn = Callable[[Record, ModelDispatcher], Record]

    def __init__(self, dispatcher: ModelDispatcher, writer: JsonlCheckpointer, process_fn: ProcessFn, concurrency: int = 32):
        self.dispatcher = dispatcher
        self.writer = writer
        self.process_fn = process_fn
        self.exec = ThreadedExecutor(concurrency)

    def _process_one(self, rec: Record):
        if self.writer.already_done(rec.get("id")):
            logger.debug("跳过 %s", rec.get("id"))
            return
        try:
            enriched = self.process_fn(rec, self.dispatcher)
            self.writer.write(enriched)
            logger.info("✅ 完成 %s", rec.get("id"))
        except Exception as e:  # noqa: BLE001
            logger.exception("❌ 处理 %s 失败: %s", rec.get("id"), e)

    def run_on_jsonl(self, input_path: os.PathLike):
        logger.info("🚀 开始处理 %s", input_path)
        with open(input_path, "r", encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        print(len(records))
        for _ in self.exec.run(self._process_one, records):
            pass

# --------------------------- 批量文件助手 ------------------------------------

def batch_process(
    input_folder: os.PathLike,
    output_folder: os.PathLike,
    api_conns: Sequence[Dict[str, str]],
    process_fn: TaskPipeline.ProcessFn,
    *,
    max_workers: int = 32,
):
    """对文件夹内所有 `*.jsonl` 执行 *process_fn*，并复用并发/断点逻辑。"""
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    dispatcher = ModelDispatcher(api_conns)

    for file in Path(input_folder).glob("*.jsonl"):
        out_path = Path(output_folder) / f"{file.stem}_QA.jsonl"
        writer = JsonlCheckpointer(out_path)
        pipe = TaskPipeline(dispatcher, writer, process_fn, concurrency=max_workers)
        pipe.run_on_jsonl(file)

    logger.info("🎉 批处理完成，结果位于 %s", output_folder)

# ---------------------------- 示例处理函数 -----------------------------------

def demo_process(rec: Dict[str, Any], dsp: ModelDispatcher) -> Dict[str, Any]:
    """示例：把 text 长度写回记录。实际使用时替换此函数。"""
    rec["text_len"] = len(rec.get("text", ""))
    return rec

# ---------------------------- 保持向后兼容 ----------------------------------
ClassifyPipeline = TaskPipeline  # 老名字别名
batch_classify = batch_process  # 老名字别名

def demo_classifier(rec, dsp):  # 旧示例函数别名
    return demo_process(rec, dsp)

# ---------------------------- CLI 入口 --------------------------------------
if __name__ == "__main__":
    import argparse, sys

    p = argparse.ArgumentParser(description="批量处理 JSONL 文件 – 通用并发框架")
    p.add_argument("input", help="包含 .jsonl 的文件夹")
    p.add_argument("output", help="输出文件夹")
    p.add_argument("--workers", type=int, default=32, help="并发线程数量")
    args = p.parse_args()

    ENDPOINTS: List[Dict[str, str]] = [
        {"api_key": "None", "base_url": "https://example.com/v1"},
    ]

    if not ENDPOINTS:
        sys.exit("❌ 请配置 ENDPOINTS")

    batch_process(
        args.input,
        args.output,
        ENDPOINTS,
        process_fn=demo_process,
        max_workers=args.workers,
    )
