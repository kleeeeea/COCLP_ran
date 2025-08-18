import os
import json
import time
import threading
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


# --------------------------------------------------------------------------
# 1. API 客户端分发器
# 这个类负责管理和轮询你的多个API终端，确保请求被均匀分配。
# --------------------------------------------------------------------------
class APIDispatcher:
    def __init__(self, api_conns):
        if not api_conns:
            raise ValueError("API connection list cannot be empty.")
        self._api_conns = api_conns
        self._lock = threading.Lock()
        self._index = 0
        self._clients = {}  # 缓存客户端实例

    def _get_client(self):
        with self._lock:
            # 轮询选择下一个 API
            conn = self._api_conns[self._index]
            self._index = (self._index + 1) % len(self._api_conns)

        base_url = conn.get("base_url")
        # 如果已经创建过这个 base_url 的客户端，直接返回
        if base_url in self._clients:
            return self._clients[base_url]

        # 否则，创建一个新的客户端并缓存
        client = OpenAI(
            api_key=conn.get("api_key", "None"),
            base_url=base_url
        )
        self._clients[base_url] = client
        print(f"Created and cached new OpenAI client for: {base_url}")
        return client


# --------------------------------------------------------------------------
# 2. ComfyUI 节点主类
# 封装了你的所有处理逻辑。
# --------------------------------------------------------------------------
class BatchProcessDocumentsNode:
    def __init__(self):
        self.valid_grades = set()
        self.valid_years = set()
        self.valid_subjects = set()
        self.valid_filetypes = set()
        self.model_name = "qwen"  # 默认模型名

    @classmethod
    def INPUT_TYPES(cls):
        current_node_dir = os.path.dirname(__file__)
        default_config_path = os.path.join(current_node_dir, "config")

        return {
            "required": {
                "input_folder": ("STRING", {"default": "", "multiline": False}),
                "output_folder": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"multiline": False}),
                "api_url": ("STRING", {"multiline": False}),
                "model_name": ("STRING", {"default": "qwen", "multiline": False}),  # ✅ 新增输入
                "valid_files_folder": ("STRING", {"default": default_config_path, "multiline": False}),
                "max_workers": ("INT", {"default": 8, "min": 1, "max": 64}),
                "trigger": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("out_path",)
    OUTPUT_NODE = True
    FUNCTION = "execute_batch_process"
    CATEGORY = "MY_NODES/Document Processing"

    # ---------------------------
    # 辅助函数 (源自你的代码)
    # ---------------------------
    def _read_from_file(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return {line.strip() for line in f if line.strip()}
        except FileNotFoundError:
            print(f"Warning: Validation file not found at {path}. Proceeding with an empty set.")
            return set()

    def _truncate_text_simple(self, text, max_length=3000):
        return text[:max_length]

    def _split_text(self, text, max_length=4000):
        return [text[i:i + max_length] for i in range(0, len(text), max_length)]

    def _extract_categories(self, text):
        result = {"学科": "其他", "文件类型": "其他", "子分类": "其他"}
        for line in text.splitlines():
            for key in result:
                if line.startswith(key + "："):
                    result[key] = line.split("：", 1)[1].strip()
        return result

    # ---------------------------
    # 模型分类函数 (源自你的代码)
    # ---------------------------
    def _classify_by_grade(self, data, client):
        completion = client.chat.completions.create(
            model=self.model_name, messages=[{'role': 'system', 'content': '''
          你是一个智能路径解析助手，能够从文件路径中提取**年级**。
### **任务要求**
1. **提取年级** - 年级信息通常包含 **高一、高二、高三** 或 **初一、初二、初三、一年级、二年级、三年级等**，并且可能多次出现在路径中。
   - 提取路径中的年级信息，例如 `高二`。
   - 如果无法提取年级的信息，则输出无
### **输出示例**
高一
五年级
无
**注意**
请严格按照格式输出，不要输出多余地解释以及说明
            '''}, {'role': 'user', 'content': data["path"]}],
            temperature=0.7, stream=False, timeout=30)
        return completion.choices[0].message.content.strip()

    def _classify_by_academic_year(self, data, client):
        completion = client.chat.completions.create(
            model=self.model_name, messages=[{'role': 'system', 'content': '''
          你是一个智能路径解析助手，能够从文件路径中提取 **学年**。
### **任务要求**
1. **提取学年** - 学年格式通常为 `XXXX-XXXX学年`（如 `2023-2024学年`）。
   - 学年信息通常位于路径中较前的位置，例如 `/2023学年/`。
   - 若显示的学年为2023学年，则输出为2023-2024学年，其他学年也按此规则
   - 如果无法提取学年的信息，则输出无
### **输出示例**
2021-2022学年
2020-2021学年
无
**注意**
请严格按照格式输出，不要输出多余地解释以及说明
'''}, {'role': 'user', 'content': data["path"]}],
            temperature=0.7, stream=False, timeout=30)
        return completion.choices[0].message.content.strip()

    def _classify_by_model(self, data, client):
        ext_chunks = self._split_text(data["text"], max_length=4000)
        valid_subjects_str = ','.join(f"'{s}'" for s in self.valid_subjects)
        valid_filetypes_str = ','.join(f"'{f}'" for f in self.valid_filetypes)

        completion = client.chat.completions.create(
            model=self.model_name, messages=[{'role': 'system', 'content': f'''
           请根据给定的文件路径和部分内容进行分析，对文件进行分类。将文件划分为以下类别：
学科：从路径中的子文件夹名称中提取学科名称，学科有 {valid_subjects_str}，如果不是上述给出的学科，输出“其他”。
文件类型：根据路径中的文件夹位置来确定文件类型，可选的类型包括 {valid_filetypes_str}，如果不是上述给出的类型，输出“其他”。
子分类：根据文件夹名称确定文件的具体分类，如果无法判断输出“其他”。
示例路径： /mnt/dataex03/task01/CM_01/教育学院版/英语/小学段/小学/基础课例/三年级/英语小学段基础课例材料长兴小学黄佳菠/3AM3U2P3教学设计——长兴小学黄佳菠.docx
输出请按照以下固定格式：
学科：英语
文件类型：教案
子分类：教学设计
注意！子分类可以没有且子分类不能是文件名, 去除学期和时间、班级、文件后缀等不相关信息，只保留分类的关键信息，不额外添加内容。
如果提取出的学科，文件类型和子分类不符合常识，则统一输出其他
'''}, {'role': 'user', 'content': "路径为：" + data["path"] + "  部分内容为：" + str(ext_chunks[0])}],
            temperature=0.7, stream=False, timeout=30)
        result = self._extract_categories(completion.choices[0].message.content)
        return result["学科"], result["文件类型"], result["子分类"]

    def _classify_by_knowledge(self, data, client):
        text_chunks = self._split_text(data["text"], max_length=7000)
        completion = client.chat.completions.create(
            model=self.model_name, messages=[{'role': 'system', 'content': '''
            你是一个智能文本分析助手，能够识别文本内容是否属于试题或试卷，并提取对应的知识点，只需要二到十个。
**任务要求：** 1. **判断文本类型**：
   - 如果文本内容是**试题或试卷**，继续执行下一步。
   - 如果文本内容**不是**试题或试卷，则输出："无"。
2. **提取知识点**：
   - 解析试题/试卷内容，并确定考察的知识点，如语文的**文言文、现代文阅读、作文**，数学的**函数、几何、概率**等。
   - 知识点需概括为清晰、具体的术语，便于分类整理。
3. **只输出关键词**：
   不要输出多余内容，关键词不超过十条
**输出格式（如果是试题/试卷）：**
"知识点1;知识点2;知识点3"
**输出格式（如果不是试题/试卷）：**
"无"
'''}, {'role': 'user', 'content': text_chunks[0]}],
            temperature=0.7, stream=False, timeout=30)
        out = completion.choices[0].message.content.strip()
        if out == "无":
            return "无"
        return "; ".join([k.strip() for k in out.split(";") if k.strip()])

    def _validate_or_retry(self, data, client, classify_fn, allowed_values, retries=1):
        for i in range(retries + 1):
            try:
                result = classify_fn(data, client).strip()
                if result in allowed_values:
                    return result
            except Exception as e:
                print(f"Attempt {i + 1} failed with error: {e}")
        return "无"

    # ---------------------------
    # 核心处理函数
    # ---------------------------
    def _process_single_record(self, record, dispatcher):
        full_text = record.get("text", "")
        short_text = self._truncate_text_simple(full_text)

        client = dispatcher._get_client()

        # 年级、学年
        record["grade"] = self._validate_or_retry({"path": record.get("path", "")}, client, self._classify_by_grade,
                                                  self.valid_grades)
        record["school_year"] = self._validate_or_retry({"path": record.get("path", "")}, client,
                                                        self._classify_by_academic_year, self.valid_years)

        # 学科、文件类型、子分类
        try:
            model_input = {"path": record.get("path", ""), "text": short_text}
            subject, ftype, subcat = self._classify_by_model(model_input, client)
            record["discipline"] = subject if subject in self.valid_subjects else "其他"
            record["file_type"] = ftype if ftype in self.valid_filetypes else "其他"
            record["subcategories"] = subcat
        except Exception as e:
            print(f"Error classifying model for {record.get('path', 'N/A')}: {e}")
            record["discipline"] = record["file_type"] = record["subcategories"] = "无"

        # 知识点
        try:
            record["knowledge_point"] = self._classify_by_knowledge({"text": full_text}, client)
        except Exception as e:
            print(f"Error classifying knowledge for {record.get('path', 'N/A')}: {e}")
            record["knowledge_point"] = "无"

        return record

    # ---------------------------
    # ComfyUI 节点执行入口
    # ---------------------------
    def execute_batch_process(self, input_folder, output_folder, api_key, api_url,
                              valid_files_folder, max_workers, trigger, model_name):

        if not trigger:
            return ("等待触发...",)

        self.model_name = model_name  # ✅ 设置当前模型名
        api_connections=json.dumps([
                {"api_key": api_key, "base_url": api_url},
                {"api_key": api_key, "base_url": api_url}
            ], indent=2)
        start_time = time.time()
        print("--- Document Batch Processing Started ---")

        # 1. 解析和验证输入
        try:
            api_conns = json.loads(api_connections)
            if not isinstance(api_conns, list) or not all(isinstance(i, dict) for i in api_conns):
                raise ValueError("API Connections must be a JSON array of objects.")
        except json.JSONDecodeError as e:
            return (f"错误: API Connections JSON 格式无效: {e}",)

        if not os.path.isdir(input_folder):
            return (f"错误: 输入文件夹不存在: {input_folder}",)

        os.makedirs(output_folder, exist_ok=True)

        # 2. 加载验证集
        self.valid_grades = self._read_from_file(os.path.join(valid_files_folder, "grade"))
        self.valid_years = self._read_from_file(os.path.join(valid_files_folder, "academic_year"))
        self.valid_subjects = self._read_from_file(os.path.join(valid_files_folder, "discipline"))
        self.valid_filetypes = self._read_from_file(os.path.join(valid_files_folder, "file_type"))
        print(
            f"Loaded {len(self.valid_grades)} grades, {len(self.valid_years)} years, {len(self.valid_subjects)} subjects, {len(self.valid_filetypes)} filetypes.")

        # 3. 初始化分发器和任务列表
        dispatcher = APIDispatcher(api_conns)
        tasks = []
        for filename in os.listdir(input_folder):
            if filename.endswith(".jsonl"):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)
                tasks.append((input_path, output_path))

        if not tasks:
            return ("未在输入文件夹中找到 .jsonl 文件。",)

        print(f"Found {len(tasks)} .jsonl files to process.")

        # 4. 使用线程池并发处理
        processed_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有文件的处理任务
            future_to_path = {}
            for input_path, output_path in tasks:
                with open(input_path, "r", encoding="utf-8") as f_in:
                    records = [json.loads(line) for line in f_in]

                self._process_file_records(records, dispatcher, output_path)
                future = executor.submit(self._process_file_records, records, dispatcher, output_path)
                future_to_path[future] = input_path

            # 获取处理结果
            for future in as_completed(future_to_path):
                input_path = future_to_path[future]
                try:
                    num_records = future.result()
                    print(f"✅ Successfully processed {input_path} ({num_records} records).")
                    processed_count += 1
                except Exception as e:
                    print(f"❌ Error processing file {input_path}: {e}")

        end_time = time.time()
        duration = end_time - start_time
        report = f"处理完成! 共处理 {processed_count}/{len(tasks)} 个文件. " \
                 f"输出已保存至: {output_folder}. 总耗时: {duration:.2f} 秒."
        print(f"--- {report} ---")

        # return (report,)
        return (output_folder,)

    def _process_file_records(self, records, dispatcher, output_path):
        processed_records = [self._process_single_record(rec, dispatcher) for rec in records]
        with open(output_path, "w", encoding="utf-8") as f_out:
            for rec in processed_records:
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return len(records)


