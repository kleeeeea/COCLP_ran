import json
import os
from pathlib import Path
import logging
from typing import List, Dict, Any
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config'))

# 添加到 sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from pathlib import Path
from batch import ModelDispatcher, TaskPipeline, JsonlCheckpointer, SmartAPIDispatcher

def split_text(text: str, maxlen: int = 2000) -> List[str]:
    return [text[i:i + maxlen] for i in range(0, len(text), maxlen)]

def extract_qa_pairs(rec: Dict[str, Any], dsp, model_name: str = "qwen") -> Dict[str, Any]:

    """
    从文本中生成多道选择题，输出为对话结构 [{"role": "user", ...}, {"role": "assistant", ...}, ...]
    """

    def parse_json_qa_to_dialogue(text: str) -> List[Dict[str, str]]:
        """解析模型 JSON Lines 输出为 role 对话格式"""
        items = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                qa = json.loads(line)
                question = qa.get("q", "").strip()
                answer = qa.get("a", "").strip()
                if question and answer:
                    items.extend([
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ])
            except json.JSONDecodeError:
                logging.warning(f"无法解析行为 JSON：{line}")
        return items

    # 基本信息
    text = rec.get("text", "")
    rec_id = rec.get("id"," ")

    if not text :
        rec["qa_pairs"] = []
        return rec

    qa_pairs = []
    for chunk in split_text(text, 2000):
        prompt = """
            你是一位擅长设计高质量选择题的老师，请根据以下内容生成多道选择题。

            要求如下：
            1. 每题包含题干、4 个选项（A~D）、标准答案和解析。
            2. 输出格式为 JSON Lines，每行一个 JSON 对象，格式如下：
            {"q": "题干+选项", "a": "答案+解析"}
            3. 不要解释说明，不要编号，不要空行，只输出纯 JSON Lines。

            请处理下面内容：
            """+chunk.strip()
        try:
            response = dsp.chat(
                messages=[
                    {"role": "system", "content": "你是一名资深出题专家"},
                    {"role": "user", "content": prompt}
                ],
                model=model_name,
                temperature=0.8,
            )
            content = response.choices[0].message.content.strip()

            parsed = parse_json_qa_to_dialogue(content)
            for i in range(0, len(parsed), 2):
                qa_pair = parsed[i:i + 2]
                if len(qa_pair) == 2 and qa_pair[0]["role"] == "user" and qa_pair[1]["role"] == "assistant":
                    if validate_qa_pair(qa_pair, dsp,model=model_name):
                        qa_pairs.extend(qa_pair)
        except Exception as e:
            logging.exception(f"处理出题失败：{e}")
            continue
    rec["qa_pairs"] = qa_pairs
    return rec

def validate_qa_pair(qa_pair: List[Dict[str, str]], dsp, model="qwen") -> bool:
    """
    使用模型验证一个选择题问答对是否合理
    """
    if len(qa_pair) != 2:
        return False

    user_msg = qa_pair[0]["content"]
    assistant_msg = qa_pair[1]["content"]

    validation_prompt = f"""
你是一位资深出题专家和审题老师。

请你帮忙判断下面这道题是否是一个**高质量的选择题**，判断维度如下：
1. 问题是否清晰、独立、可理解；
2. 选项是否语言自然、无歧义；
3. 答案是否准确，且解析有合理的逻辑支撑；
4. 不应包含胡编的内容；
5. 不应有结构错误、语病或重复内容。
6. 不应该出现本文、本单元、本书、第几段、第几行、前文这类指代不明确的提问

请你最终只返回“合理”或“不合理”。

下面是待验证的问答对：

题目与选项：
{user_msg}

答案与解析：
{assistant_msg}
""".strip()

    try:
        response = dsp.chat(
            messages=[
                {"role": "system", "content": "你是资深出题审核专家"},
                {"role": "user", "content": validation_prompt}
            ],
            model=model,
            temperature=0.8
        )
        reply = response.choices[0].message.content.strip()
        return "合理" in reply and "不合理" not in reply
    except Exception as e:
        logging.exception(f"验证 QA 对出错: {e}")
        return False

class BatchGenerateQANode:
    """
    批量生成多道选择题，并分片输出
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": ("STRING", {"multiline": False, "default": "", "lazy": True}),
                "output_folder": ("STRING", {"multiline": False, "default": "", "lazy": True}),
                "api_key": ("STRING", {"multiline": False}),
                "api_url": ("STRING", {"multiline": False}),
                "model_name": ("STRING", {"default": "qwen", "multiline": False}),  # ✅ 新增
                "max_workers": ("INT", {"default": 24, "min": 1, "max": 128}),
                "shard_size": ("INT", {"default": 10000, "min": 0, "max": 100000})
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("qa_jsonl_path", "shard_folder")
    OUTPUT_NODE = True
    FUNCTION = "process"

    CATEGORY = "MY_NODES/BatchGenerateQA"

    def process(self, input_folder, output_folder, api_key, api_url, max_workers, shard_size, model_name):

        api_conns=json.dumps([
                {"api_key": api_key, "base_url": api_url},
                {"api_key": api_key, "base_url": api_url}
            ], indent=2)
        
        # Parse api_conns
        try:
            api_conns_list = json.loads(api_conns)
        except Exception as e:
            raise ValueError(f"无法解析api_conns，请输入JSON格式列表。错误：{e}")

        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        dispatcher = SmartAPIDispatcher(api_conns_list)

        results = []

        for file in input_folder.glob("*.jsonl"):
            logger = logging.getLogger(__name__)
            logger.info(f"🚀 开始处理文件: {file.name}")

            output_path = output_folder / f"{file.stem}_qa.jsonl"
            flat_output_dir = output_folder / f"{file.stem}_flat"
            flat_output_dir.mkdir(exist_ok=True, parents=True)

            writer = JsonlCheckpointer(str(output_path), "id", record_dir=output_folder)
            pipeline = TaskPipeline(dispatcher, writer,  lambda rec, dsp: extract_qa_pairs(rec, dsp, model_name), concurrency=max_workers)
            pipeline.run_on_jsonl(str(file))

            self._flatten_and_shard(str(output_path), str(flat_output_dir), shard_size)

            logger.info(f"✅ {file.name} 扁平化完成，输出目录: {flat_output_dir}")

            results.append((str(output_path), str(flat_output_dir)))

        # 只返回第一个结果（一个节点只能返回固定数量的输出）
        if results:
            return results[0]
        else:
            return ("", "")

    def _flatten_and_shard(self, input_jsonl: str, output_dir: str, shard_size: int = 10000):
        os.makedirs(output_dir, exist_ok=True)
        shard_idx = 0
        qa_count = 0
        out_f = None

        def open_new_file():
            nonlocal shard_idx, out_f
            if out_f:
                out_f.close()
            out_path = os.path.join(output_dir, f"qa_shard_{shard_idx:04d}.jsonl")
            out_f = open(out_path, "w", encoding="utf-8")
            shard_idx += 1
            return out_f

        out_f = open_new_file()

        with open(input_jsonl, "r", encoding="utf-8") as fin:
            for line in fin:
                rec = json.loads(line)
                id_val = rec.get("id")
                qas = rec.get("qa_pairs", [])
                for i in range(0, len(qas), 2):
                    qa_pair = qas[i:i + 2]
                    if len(qa_pair) == 2 and qa_pair[0].get("role") == "user" and qa_pair[1].get("role") == "assistant":
                        out_rec = {"id": id_val, "qa_pair": qa_pair}
                        out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                        qa_count += 1
                        if qa_count % shard_size == 0:
                            out_f = open_new_file()
        if out_f:
            out_f.close()





