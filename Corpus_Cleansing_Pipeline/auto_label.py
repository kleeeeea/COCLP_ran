import os
import random
import jsonlines
import re
import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import threading
from tqdm import tqdm
import sys


def reservoir_sample_jsonl(input_dir, output_file, k):
    """
    使用蓄水池抽样算法从目录下的所有 .jsonl 文件中抽样。

    Args:
        input_dir (str): 包含 .jsonl 文件的目录。
        output_file (str): 输出的 .jsonl 文件路径。
        k (int): 要抽样的项目数量。
    """
    reservoir = []
    count = 0  # 总读取行数

    # 检查输入目录是否存在
    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录不存在于 {input_dir}")
        return

    print(f"开始从 '{input_dir}' 进行水塘抽样...")
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.jsonl'):
                file_path = os.path.join(root, file)
                try:
                    with jsonlines.open(file_path, mode='r') as reader:
                        for obj in reader:
                            count += 1
                            if len(reservoir) < k:
                                reservoir.append(obj)
                            else:
                                r = random.randint(0, count - 1)
                                if r < k:
                                    reservoir[r] = obj
                except Exception as e:
                    print(f"读取文件 {file_path} 时出错: {e}")

    print(f"总共处理行数: {count}")
    print(f"抽样行数: {len(reservoir)}")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(reservoir)
    print(f"抽样数据已保存至 {output_file}")


# --- 全局配置 ---
# 提示词模板 - 结合质量判断和教育属性判断
PROMPT_TEMPLATE = r'''## 角色 
你是一位教育学科领域专家，熟悉各学段学生的发展特点与教育目标。

## 任务 
对给定的**选择题问题文本（Q）+答案与解析（A）**，从以下三个判断依据出发，进行**教育属性分类**：

> **判断依据**（满足**任一条件**即分类为1）： 
> 1. 该问题是否属于教育领域知识范畴（包括但不限于学科知识、技能培养、素养发展等）？
> 2. 该问题是否适合出现在学生的考试或学习评估中，或可作为教学材料使用？
> 3. 该问题的解答是否有助于知识传递、能力培养或素养发展？

### 分类选项（二选一）：
- 教育属性内容：1
- 非教育属性内容：0

### 注意事项：
- 只要满足**任一**判断依据即可分类为1
- 教育领域知识范畴包括所有可能被教授或学习的知识
- 不要求题目明确面向学生，只要内容具有教育潜力即可
- 知识传递包括直接和间接的知识考查
- 能力培养和素养发展包括各种认知和技能发展

## Input 
Question: {question}
Answer：{answer}

## Output
最终以如下JSON格式返回结果：
{{
    "reason": "你的分析过程（说明满足的判断依据，不超过150字）",
    "label": 0/1
}}'''

# 线程局部存储，用于每个线程维护一个OpenAI客户端实例
_thread_local = threading.local()


# --- 辅助函数 ---
def get_openai_client(api_configs: list):
    """
    获取一个OpenAI客户端实例。
    每个线程会维护一个客户端实例，并从提供的API配置中随机选择一个。
    """
    if not hasattr(_thread_local, "client"):
        config = random.choice(api_configs)
        _thread_local.client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
    return _thread_local.client


def call_llm(text: str, api_configs: list, temperature: float = 0.1, top_p: float = 0.1):
    """
    调用大语言模型（LLM）进行文本生成。
    """
    client = get_openai_client(api_configs)
    try:
        response = client.chat.completions.create(
            model="qwen",
            messages=[{"role": "user", "content": text}],
            temperature=temperature,
            top_p=top_p,
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"调用LLM时出错: {e}")
        return None


def extract_json_from_text(text: str):
    """
    从文本中提取最外层的JSON对象。
    """
    if not text:
        return None
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def remove_markers_from_text(text: str, start_marker: str = "<think>", end_marker: str = "</think>"):
    """
    移除文本中指定标记及其之间的内容（例如LLM思考过程）。
    """
    if not text:
        return ""
    pattern = re.escape(start_marker) + '.*?' + re.escape(end_marker)
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text


# --- 主要处理逻辑 ---
def process_single_item(item: dict, prompt_template: str, api_configs: list, lock: threading.Lock,
                        writer: jsonlines.Writer):
    """
    处理单个数据项，并将结果写入单个文件。
    """
    original_item_id = item.get("id", "N/A")
    try:
        qa_pair = item.get("qa_pair")
        question = qa_pair[0]['content']
        answer = qa_pair[1]['content']
        # 格式化Prompt
        formatted_prompt = prompt_template.format(question=question, answer=answer)
    except Exception as e:
        print(f"准备项目 {original_item_id} 时出错: {e}")
        # 出现任何程序性失败，直接丢弃该数据，不写入文件
        return

    try:
        # 调用LLM进行一次性判断
        llm_response_str = call_llm(formatted_prompt, api_configs)
        # 移除LLM的思考标记并提取JSON
        cleaned_response = remove_markers_from_text(llm_response_str)
        llm_result_json = extract_json_from_text(cleaned_response)

        if llm_result_json and 'label' in llm_result_json:
            copy_item = item.copy()
            copy_item['label'] = llm_result_json.get('label')
            # 将处理后的字典写入文件
            with lock:
                writer.write(copy_item)
        else:
            print(f"警告: 无法为项目 {original_item_id} 提取有效标签。已跳过。")

    except Exception as e:
        print(f"LLM处理项目 {original_item_id} 时出错: {e}")


def process_data_batch(input_file_path: str, output_file_path: str, num_workers: int, prompt_template: str,
                       api_configs: list):
    """
    批量处理数据，使用线程池并行执行。
    """
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    lock = threading.Lock()  # 用于线程安全地写入文件
    print(f"从以下路径读取数据: {input_file_path}")
    data_to_process = []
    try:
        with jsonlines.open(input_file_path) as reader:
            for item in reader:
                data_to_process.append(item)
    except FileNotFoundError:
        print(f"错误: 输入文件未找到于 {input_file_path}")
        return
    except Exception as e:
        print(f"读取输入文件 {input_file_path} 时出错: {e}")
        return

    total_items = len(data_to_process)
    print(f"待处理项目总数: {total_items}")
    print(f"输出将保存至: {output_file_path}")
    print(f"使用 {num_workers} 个工作线程开始批量处理...")

    with jsonlines.open(output_file_path, mode='w') as writer:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务到线程池
            futures = [executor.submit(process_single_item, item, prompt_template, api_configs, lock, writer) for item
                       in data_to_process]
            # 使用tqdm显示进度
            for future in tqdm(futures, total=total_items, desc="标注项目"):
                future.result()  # 确保所有任务完成，并捕获可能的异常

    print("批量处理完成。")


# ====================================================================================================
# ComfyUI Node Definitions (ComfyUI 节点定义)
# ====================================================================================================

class SampleDataNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_dir": ("STRING", {"default": "./input_data"}),
                "output_file": ("STRING", {"default": "./sampled_output.jsonl"}),
                "sample_k": ("INT", {"default": 10000, "min": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "sample_data"
    CATEGORY = "MY_NODES/LLM"

    def sample_data(self, input_dir, output_file, sample_k):
        reservoir_sample_jsonl(input_dir, output_file, sample_k)
        return (f"从 {input_dir} 抽样 {sample_k} 个项目到 {output_file} 完成。",)


class LabelDataNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING", {"default": "./sampled_output.jsonl"}),
                "output_file": ("STRING", {"default": "./labeled_output.jsonl"}),
                "base_urls": (
                "STRING", {"default": "http://127.0.0.1:8000/v1", "multiline": True, "placeholder": "每行一个URL"}),
                "api_key": ("STRING", {"default": "None"}),
                "num_workers": ("INT", {"default": 8, "min": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "label_data"
    CATEGORY = "MY_NODES/LLM"

    def label_data(self, input_file, output_file, base_urls, api_key, num_workers):
        # 从输入URL创建API配置
        url_list = [url.strip() for url in base_urls.split('\n') if url.strip()]
        if not url_list:
            return ("错误: 未提供有效的 base URL。",)

        api_configs = [{"api_key": api_key, "base_url": url} for url in url_list]

        process_data_batch(input_file, output_file, num_workers, PROMPT_TEMPLATE, api_configs)
        return (f"已标注来自 {input_file} 的数据并保存到 {output_file}。",)


# ====================================================================================================
# Node Mappings for ComfyUI (ComfyUI 节点映射)
# ====================================================================================================

