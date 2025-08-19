import os
import glob
import multiprocessing
import jsonlines
import unicodedata
import regex as re
from functools import partial
from tqdm import tqdm


## 答题卡过滤器
class AnswerSheetFilter:
    """过滤答题卡"""
    def __init__(self):
        self.keywords = ["答题纸", "答题卡", "答题卷", "答卷"]

    def process(self, json_obj):
        ITEM_PATH = "path"  # 根据你的数据结构调整字段名
        path = json_obj.get(ITEM_PATH, '')
        filename = os.path.basename(path)

        is_answer_sheet = False
        for keyword in self.keywords:
            if keyword in filename:
                is_answer_sheet = True
        if "解答卷" in filename:
            is_answer_sheet = False

        if is_answer_sheet:
            return None  # 过滤掉
        else:
            return json_obj  # 保留

## 短文本过滤器
class ShortLengthFilter:
    """过滤有效字符少于20的短文本"""

    @staticmethod
    def count_meaningful_chars(text: str) -> int:
        # 匹配所有字母类字符（不包含数字、标点、空白等）
        pattern = r"\p{Letter}"

        # "Hello, 世界！こんにちは！안녕하세요~ 123 Привет! αβγ 标点！"
        # 28

        meaningful_chars = re.findall(pattern, text, re.UNICODE)
        return len(meaningful_chars)

    def __init__(self, minLength = 100):
        self.minLength = minLength

    def process(self, json_obj):
        ITEM_TEXT = 'text'
        # if self.count_meaningful_chars(json_obj[ITEM_TEXT]) >= self.minLength:
        #     return json_obj
        # return None
        num=self.count_meaningful_chars(json_obj[ITEM_TEXT])
        if num >= self.minLength:
            return json_obj
        return None


class IntegratedPruner:
    # Markdown 清洗相关属性
    img_pat = re.compile(r'!\[([^\n\r]*?)\]\(([^()\s]+)?\)')
    url_pat = re.compile(r'\[([^\n\r]*?)\]\(([^()\s]+)?\)')
    error_img_pat = re.compile(r'!\[([^\n\r]*?)\]')
    markdown_repl = ''

    # 邮箱清洗相关属性
    email_pat = re.compile(r'[A-Za-z0-9.\-+_]+@[a-z0-9.\-+_]+\.[a-z]+', re.IGNORECASE)
    email_repl = ''

    # 链接清洗相关属性
    link_pat = re.compile(
        r'(?i)\b(?:https?|ftp):\/\/'
        r'(?:[\w-]+(?:\.[\w-]+)+|localhost|(?:\/[^\s]*)?)'
        r'(?:\:[0-9]+)?(?:\/\S*)?',
        flags=re.DOTALL
    )
    link_repl = ''

    # IP 清洗相关属性
    ipv4_pat = re.compile(
        r'\b(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b')
    ipv6_pat = re.compile(
        r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|\b(?:[0-9a-fA-F]{1,4}:){1,7}:\b|\b(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}\b|\b(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}\b|\b(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}\b|\b(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}\b|\b(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}\b|\b[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})\b|\b:(?:(?::[0-9a-fA-F]{1,4}){1,7})\b')
    ip_repl = ''

    # 控制字符清洗相关属性
    @staticmethod
    def clean_control_chars(text: str) -> str:
        return ''.join(ch for ch in text if unicodedata.category(ch) != 'Cc' or ch == '\n')

    # 重复空格清洗相关属性
    space_pat = re.compile(r' {2,}')
    tab_pat = re.compile(r'\t{2,}')
    mixed_pat = re.compile(r'([ \t\n]{2,})')
    space_repl = ' '
    tab_repl = '\t'
    mixed_repl = '\n'

    # 标点截断清洗相关属性
    def __init__(self):
        self.chinese_punctuation = "。！？"
        self.english_punctuation = ".?!"
        self.all_punctuation = self.chinese_punctuation + self.english_punctuation

    def end_clip(self, text: str) -> str:
        if not text or text[-1] in self.all_punctuation:
            return text

        match = re.search(r'[{}]'.format(re.escape(self.all_punctuation)), text[::-1])
        return text[:len(text) - match.start()] if match else ""

    # 圆圈数字清洗相关属性
    circled_pat = re.compile(r'\$\s*\\textcircled\s*\{\s*([^}]*)\s*\}\s*\$')
    circled_repl = ''

    @classmethod
    def clean_circled_links(cls, text: str) -> str:
        def replace_match(match):
            content = match.group(1).strip()
            return f"{content}." if content.isdigit() else cls.circled_repl

        return re.sub(cls.circled_pat, replace_match, text)

    # 文件名清洗相关属性
    suffix_list = [
        'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp', 'svg', 'mp4', 'avi', 'mov',
        'wmv', 'flv', 'mkv', 'webm', 'mpeg', 'mpg', 'mp3', ".wav", ".aac", ".flac",
        ".ogg", ".m4a", ".wma", ".alac", 'doc', 'docx', 'pdf', 'txt', 'md', 'rtf',
        'odt', 'csv', 'xls', 'xlsx', 'ods', 'ppt', 'pptx', 'odp', 'zip', 'rar',
        '7z', 'tar', 'gz', 'bz2',
    ]
    suffix_pattern = '|'.join([re.escape(ext) for ext in suffix_list])
    file_pat = re.compile(rf'([a-zA-Z0-9\[\]!@#$%^&*()_<>\/\?,+\-=]+)\.({suffix_pattern})\)?')
    file_repl = ''

    # 重复字符清洗相关属性
    latex_special_chars = r'\{}$&%#^_~<>():;@[]-|'
    excluded_chars = re.escape(latex_special_chars)
    repeat_pat_str = fr"(?:[^\p{{L}}\p{{N}}](?![{excluded_chars}])){{8,}}"
    repeat_pat = re.compile(repeat_pat_str)

    # HTML 标签清洗相关属性
    html_patterns = {
        tag: re.compile(rf'<\/?\s*{tag}\s*>')
        for tag in [
            'html', 'head', 'title', 'base', 'link', 'meta', 'style', 'script',
            'body', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'nav', 'header', 'footer',
            'main', 'section', 'article', 'aside', 'p', 'br', 'hr', 'blockquote',
            'pre', 'code', 'a', 'ul', 'ol', 'li', 'dl', 'dt', 'dd', 'table', 'thead',
            'tbody', 'tfoot', 'tr', 'th', 'td', 'caption', 'colgroup', 'col', 'form',
            'input', 'textarea', 'button', 'select', 'option', 'optgroup', 'label',
            'fieldset', 'legend', 'div', 'span', 'img', 'audio', 'video', 'source',
            'track', 'canvas', 'svg', 'math', 'template', 'noscript'
        ]
    }
    html_repl = ''

    def clean_markdown_links(self, text: str) -> str:
        text = re.sub(self.img_pat, self.markdown_repl, text)
        text = re.sub(self.url_pat, self.markdown_repl, text)
        text = re.sub(self.error_img_pat, self.markdown_repl, text)
        return text.strip()

    def clean_email(self, text: str) -> str:
        return re.sub(self.email_pat, self.email_repl, text).strip()

    def clean_links(self, text: str) -> str:
        return re.sub(self.link_pat, self.link_repl, text).strip()

    def clean_ips(self, text: str) -> str:
        text = re.sub(self.ipv4_pat, self.ip_repl, text)
        text = re.sub(self.ipv6_pat, self.ip_repl, text)
        return text.strip()

    def clean_repeat_spaces(self, text: str) -> str:
        text = re.sub(self.space_pat, self.space_repl, text)
        text = re.sub(self.tab_pat, self.tab_repl, text)
        text = re.sub(self.mixed_pat, self.mixed_repl, text)
        return text.strip()

    def clean_files(self, text: str) -> str:
        return re.sub(self.file_pat, self.file_repl, text)

    def clean_repeats(self, text: str) -> str:
        return self.repeat_pat.sub('', text)

    def clean_html_tags(self, text: str) -> str:
        for tag, pattern in self.html_patterns.items():
            text = pattern.sub(self.html_repl, text)
        return text

    def process(self, json_obj: dict,
                enable_markdown=False, enable_email=False, enable_link=False, enable_ip=False,
                enable_control_char=False, enable_repeat_space=False, enable_punctuation_clip=False,
                enable_text_circled=False, enable_filename=False, enable_repeat_char=False, enable_html=False):
        original_text = json_obj.get('text', "")
        if enable_markdown:
            original_text = self.clean_markdown_links(original_text)
        if enable_email:
            original_text = self.clean_email(original_text)
        if enable_link:
            original_text = self.clean_links(original_text)
        if enable_ip:
            original_text = self.clean_ips(original_text)
        if enable_control_char:
            original_text = self.clean_control_chars(original_text)
        if enable_repeat_space:
            original_text = self.clean_repeat_spaces(original_text)
        if enable_punctuation_clip:
            original_text = self.end_clip(original_text)
        if enable_text_circled:
            original_text = self.clean_circled_links(original_text)
        if enable_filename:
            original_text = self.clean_files(original_text)
        if enable_repeat_char:
            original_text = self.clean_repeats(original_text)
        if enable_html:
            original_text = self.clean_html_tags(original_text)
        json_obj['text'] = original_text
        return json_obj


def process_single_file(json_file_path, pruner, output_path,
                        enable_markdown=False, enable_email=False, enable_link=False, enable_ip=False,
                        enable_control_char=False, enable_repeat_space=False, enable_punctuation_clip=False,
                        enable_text_circled=False, enable_filename=False, enable_repeat_char=False, enable_html=False,
                        enable_answer_sheet_filter=False, min_text_length=100):
    try:
        os.makedirs(output_path, exist_ok=True)
        output_file_path = os.path.join(output_path, os.path.basename(json_file_path))

        answer_sheet_filter = AnswerSheetFilter()
        short_length_filter = ShortLengthFilter(minLength=min_text_length)

        with jsonlines.open(json_file_path) as reader:
            processed_objects = []
            for obj in reader:
                if not obj:
                    continue
                # 答题卡过滤
                if enable_answer_sheet_filter:
                    obj = answer_sheet_filter.process(obj)
                    if obj is None:
                        continue
                # 短文本过滤
                obj = short_length_filter.process(obj)
                if obj is None:
                    continue
                processed_obj = pruner.process(obj,
                                               enable_markdown, enable_email, enable_link, enable_ip,
                                               enable_control_char, enable_repeat_space, enable_punctuation_clip,
                                               enable_text_circled, enable_filename, enable_repeat_char, enable_html)
                processed_objects.append(processed_obj)

        with jsonlines.open(output_file_path, mode='w') as writer:
            writer.write_all(processed_objects)

    except Exception as e:
        print(f"处理文件 {json_file_path} 时出错: {e}")


def process_all_files(input_dir, output_dir, pruner, num_processes=None,
                      enable_markdown=False, enable_email=False, enable_link=False, enable_ip=False,
                      enable_control_char=False, enable_repeat_space=False, enable_punctuation_clip=False,
                      enable_text_circled=False, enable_filename=False, enable_repeat_char=False, enable_html=False,
                      enable_answer_sheet_filter=False, min_text_length=100):
    jsonl_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    if not jsonl_files:
        print(f"在 {input_dir} 中未找到 .jsonl 文件")
        return

    # 启用多进程
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    from os.path import exists
    # if exists(output_dir) and len(os.listdir(output_dir)) >= len(jsonl_files):
    #     print(f"输出目录 {output_dir} 已存在且文件数量大于等于输入文件数量，无需处理")
    #     return
    # Use tqdm for progress tracking in the main process
    with multiprocessing.Pool(processes=num_processes) as pool:
        process_func = partial(process_single_file, pruner=pruner, output_path=output_dir, enable_markdown=enable_markdown, enable_email=enable_email, enable_link=enable_link,
                            enable_ip=enable_ip, enable_control_char=enable_control_char,
                            enable_repeat_space=enable_repeat_space, enable_punctuation_clip=enable_punctuation_clip,
                            enable_text_circled=enable_text_circled, enable_filename=enable_filename,
                            enable_repeat_char=enable_repeat_char, enable_html=enable_html,
                            enable_answer_sheet_filter=enable_answer_sheet_filter,
                            min_text_length=min_text_length)
        process_func(jsonl_files[0])
        list(tqdm(pool.imap(process_func, jsonl_files), total=len(jsonl_files), desc=f"Pruning files"))

    print(f"处理完成。结果保存至 {output_dir}")


class MultiPruner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_dir": ("STRING", {
                    "default": "./input",
                    "description": "输入JSONL文件所在目录",
                    "multiline": False
                }),
                "output_dir": ("STRING", {
                    "default": "./output",
                    "description": "清洗后文件输出目录",
                    "multiline": False
                }),
                "num_processes": ("INT", {
                    "default": 1,  # 默认使用单进程避免多进程问题
                    "min": 1,
                    "description": "并行进程数（多进程可能导致某些环境下出错，建议使用1）"
                }),
                "min_text_length": ("INT", {
                    "default": 100,
                    "min": 1,
                    "description": "最短文本字符数"
                })
            },
            "optional": {
                "enable_markdown": ("BOOLEAN", {
                    "default": False,
                    "description": "启用Markdown语法清洗"
                }),
                "enable_email": ("BOOLEAN", {
                    "default": False,
                    "description": "启用邮箱清洗"
                }),
                "enable_link": ("BOOLEAN", {
                    "default": False,
                    "description": "启用链接清洗"
                }),
                "enable_ip": ("BOOLEAN", {
                    "default": False,
                    "description": "启用IP地址清洗"
                }),
                "enable_control_char": ("BOOLEAN", {
                    "default": False,
                    "description": "启用控制字符清洗"
                }),
                "enable_repeat_space": ("BOOLEAN", {
                    "default": False,
                    "description": "启用重复空格清洗"
                }),
                "enable_punctuation_clip": ("BOOLEAN", {
                    "default": False,
                    "description": "启用标点截断清洗"
                }),
                "enable_text_circled": ("BOOLEAN", {
                    "default": False,
                    "description": "启用圆圈数字清洗"
                }),
                "enable_filename": ("BOOLEAN", {
                    "default": False,
                    "description": "启用文件名清洗"
                }),
                "enable_repeat_char": ("BOOLEAN", {
                    "default": False,
                    "description": "启用重复字符清洗"
                }),
                "enable_html": ("BOOLEAN", {
                    "default": False,
                    "description": "启用HTML标签清洗"
                }),
                "enable_answer_sheet_filter": ("BOOLEAN", {
                    "default": False,
                    "description": "启用答题卡过滤"
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("输出目录路径",)
    FUNCTION = "run_pruners"
    CATEGORY = "MY_NODES/Multi_Pruner"
    OUTPUT_NODE = True

    def run_pruners(self, input_dir, output_dir, num_processes, min_text_length,
                    enable_markdown, enable_email, enable_link, enable_ip, enable_control_char,
                    enable_repeat_space, enable_punctuation_clip, enable_text_circled,
                    enable_filename, enable_repeat_char, enable_html, enable_answer_sheet_filter):
        pruner = IntegratedPruner()
        process_all_files(input_dir, output_dir, pruner, num_processes=num_processes,
                          enable_markdown=enable_markdown, enable_email=enable_email, enable_link=enable_link,
                          enable_ip=enable_ip, enable_control_char=enable_control_char,
                          enable_repeat_space=enable_repeat_space, enable_punctuation_clip=enable_punctuation_clip,
                          enable_text_circled=enable_text_circled, enable_filename=enable_filename,
                          enable_repeat_char=enable_repeat_char, enable_html=enable_html,
                          enable_answer_sheet_filter=enable_answer_sheet_filter,
                          min_text_length=min_text_length)
        print(f"已完成清洗，输入目录: {input_dir}，输出目录: {output_dir}")
        return (output_dir,)


