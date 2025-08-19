# tests/test_integrated_pruner.py

from custom_nodes.Corpus_Cleansing_Pipeline.multipruner_node_v2 import IntegratedPruner


# 确保可以导入被测类（按你的实际路径调整）


pruner =  IntegratedPruner()


# ---------- Markdown/链接/邮箱/文件/HTML ----------

def main_clean_markdown_links():
    text = "A ![img](http://a.com/x.png) B [go](https://b.io) C ![broken]"
    out = pruner.clean_markdown_links(text)
    # 所有 markdown 链接与图片被清空，保留其他文本与空白的自然结果
    assert out == "A  B  C"
def main_clean_email():
    text = "Contact me at a.b+z@test.COM! Or at foo_bar@sub.example.org."
    out = pruner.clean_email(text)
    assert "test.COM" not in out and "example.org" not in out
    assert out.strip().startswith("Contact me at")
    assert out.strip().endswith(".")
def main_clean_links():
    text = "See https://example.com and ftp://host/file and http://a/b?c=d"
    out = pruner.clean_links(text)
    assert "http" not in out and "ftp" not in out
    assert out == "See  and  and"
def main_linkpat():
    import re

    import regex
    link_pat = regex.compile(
        r'(?i)\b(?:https?|ftp):\/\/'
        r'(?:[\w-]+(?:\.[\w-]+)+|localhost|(?:\/[^\s]*)?)'
        r'(?:\:[0-9]+)?(?:\/\S*)?',
        flags=re.DOTALL
    )

    tests = [
        "http://example.com",
        "https://sub.domain.org/path/to/page",
        "ftp://files.server.net:21/download.zip",
        "http://localhost:8000/test",
        "https://EXAMPLE.Com/Case/Insensitive",
        "invalid://example.com",  # ❌ 不匹配
            "ftp://host/file",
    ]

    for t in tests:
        m = link_pat.search(t)
        print(f"{t} -> {m.group(0) if m else '不匹配'}")
def main_clean_ips():
    text = "ipv4=192.168.0.1; ipv6=fe80::1ff:fe23:4567:890a; ok"
    out = pruner.clean_ips(text)
    assert "192.168.0.1" not in out
    assert "fe80::1ff:fe23:4567:890a" not in out
    assert out.endswith("ok")

    import re

    ipv6_pat = re.compile(
        r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|"
        r"\b(?:[0-9a-fA-F]{1,4}:){1,7}:\b|"
        r"\b(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}\b|"
        r"\b(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}\b|"
        r"\b(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}\b|"
        r"\b(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}\b|"
        r"\b(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}\b|"
        r"\b[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})\b|"
        r"\b:(?:(?::[0-9a-fA-F]{1,4}){1,7})\b"
    )

    tests = [
        "2001:0db8:85a3:0000:0000:8a2e:0370:7334",   # 全 8 段
        "2001:db8:85a3::8a2e:370:7334",              # 中间压缩
        "2001:db8::",                                 # 尾部压缩
        "::1",                                        # 回环地址
        "::",                                         # 全零地址
        "fe80::a6db:30ff:fe98:e946",                  # 链路本地
        "2001:db8:0:0:0:0:2:1",                       # 等价于 2001:db8::2:1
        "2001:db8:::1",                               # ❌ 非法，多个冒号处错误
        "gggg::1",                                    # ❌ 非法，非十六进制
        "12345::",                                    # ❌ 非法，段超过4位
    ]

    for t in tests:
        m = ipv6_pat.search(t)
        print(f"{t:40} -> {'匹配' if m else '不匹配'}")

def main_clean_files():
    # 注意：clean_files 不 strip，按实现只做替换
    text = "report.pdf) and pic.jpg plus audio.wav and data.csv"
    out = pruner.clean_files(text)
    for name in ("report.pdf", "pic.jpg", "audio.wav", "data.csv"):
        assert name not in out
    # 至少验证主要内容被清走
    assert "and" in out

    import re

    suffix_list = [
        'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp', 'svg', 'mp4', 'avi', 'mov',
        'wmv', 'flv', 'mkv', 'webm', 'mpeg', 'mpg', 'mp3', ".wav", ".aac", ".flac",
        ".ogg", ".m4a", ".wma", ".alac", 'doc', 'docx', 'pdf', 'txt', 'md', 'rtf',
        'odt', 'csv', 'xls', 'xlsx', 'ods', 'ppt', 'pptx', 'odp', 'zip', 'rar',
        '7z', 'tar', 'gz', 'bz2',
    ]
    suffix_pattern = '|'.join([re.escape(ext) for ext in suffix_list])
    file_pat = re.compile(rf'([a-zA-Z0-9\[\]!@#$%^&*()_<>\/\?,+\-=]+)\.({suffix_pattern})\)?')

    tests = [
        "report.pdf",
        "image.png)",
        "archive.tar.gz",
        "music.mp3",
        "weird_file-name@2024.docx",
        "not_a_file.word",
    ]

    for t in tests:
        m = file_pat.search(t)
        print(f"{t:25} -> {m.groups() if m else '不匹配'}")
def main_clean_html_tags():
    text = "<div>Hello</div><p>World</p><span>!</span>"
    out = pruner.clean_html_tags(text)
    assert out == "HelloWorld!"

    import re

    # 构造一个 demo，只取几个标签来演示
    tags = ['div', 'p', 'br']
    html_patterns = {tag: re.compile(rf'<\/?\s*{tag}\s*>') for tag in tags}

    tests = [
        "<div>Hello</div>",
        "<p>Paragraph</p>",
        "Line break<br>Next line",
        "<div class='x'>Not match</div>",  # 含属性的情况
    ]

    for t in tests:
        print(f"原文: {t}")
        for tag, pat in html_patterns.items():
            if pat.search(t):
                print(f"  匹配到 <{tag}> 或 </{tag}>")
        print("-" * 40)
# ---------- 空白/控制字符/重复 ----------

def main_clean_repeat_spaces():
    # 两个空格 -> 一个空格；两个制表符 -> 一个制表符；2+ 的混合空白 -> 换行
    text = "a  b\t\tc\n\nd"
    out = pruner.clean_repeat_spaces(text)
    # 期望：空格合并、制表符合并、连续换行压成一个换行
    assert out == "a b\tc\nd"
def main_clean_control_chars_keeps_newline():
    s = "ok\nbad\x00good\x01"
    out = IntegratedPruner.clean_control_chars(s)
    # \x00, \x01（Cc 类别）应被清除，但 \n 保留
    assert out == "ok\ngood"
# ---------- 标点截断 ----------

def main_end_clip_with_punctuation_unchanged():
    assert pruner.end_clip("Hello world.") == "Hello world."
    assert pruner.end_clip("中文句子。") == "中文句子。"
    import re

    class Demo:
        def __init__(self):
            self.chinese_punctuation = "。！？"
            self.english_punctuation = ".?!"
            self.all_punctuation = self.chinese_punctuation + self.english_punctuation

        def end_clip(self, text: str) -> str:
            if not text or text[-1] in self.all_punctuation:
                return text
            match = re.search(r'[{}]'.format(re.escape(self.all_punctuation)), text[::-1])
            return text[:len(text) - match.start()] if match else ""

    demo = Demo()

    tests = [
        "今天下雨了。",            # 已有中文句号
        "This is a test?",       # 已有英文问号
        "你好，这是个测试没有标点", # 没有终止符号
        "Hello world! extra",    # 末尾没有标点，但中间有
        "完全没有标点",             # 整个字符串没有标点
    ]

    for t in tests:
        print(f"原文: {t} -> 处理后: {demo.end_clip(t)}")
def main_end_clip_without_punctuation_returns_to_last_one():
    # 无终止标点则回溯到最近一次出现；若完全没有则返回空串
    assert pruner.end_clip("Hi! still running") == "Hi!"
    assert pruner.end_clip("没有终止符号与句点") == ""

# ---------- 圆圈数字 ----------

def main_clean_circled_links_digits_to_number_dot():
    s = r"Step $ \textcircled{1} $ then $ \textcircled{ 2 } $ done."
    out = pruner.clean_circled_links(s)
    assert out == "Step 1. then 2. done."
def main_clean_circled_links_non_digit_removed():
    s = r"Bad $ \textcircled{ A } $ ignored."
    out = pruner.clean_circled_links(s)
    assert out == "Bad  ignored."

    import re

    class Demo:
        circled_pat = re.compile(r'\$\s*\\textcircled\s*\{\s*([^}]*)\s*\}\s*\$')
        circled_repl = ''

        @classmethod
        def clean_circled_links(cls, text: str) -> str:
            def replace_match(match):
                content = match.group(1).strip()
                return f"{content}." if content.isdigit() else cls.circled_repl
            return re.sub(cls.circled_pat, replace_match, text)


    tests = [
        "Step $ \\textcircled{1} $ do this",
        "Then $\\textcircled{2}$ next",
        "Bad $ \\textcircled{A} $ ignored",
        "Mixed $ \\textcircled{3} $ and $ \\textcircled{B} $",
    ]

    for t in tests:
        print(f"原文: {t}")
        print(f"清洗后: {Demo.clean_circled_links(t)}")
        print("-" * 40)
# ---------- 重复符号清洗（依赖 regex 才测） ----------

def main_clean_repeats_with_regex_backend():
    import regex  # type: ignore

    s = "hello " + "-" * 12 + " world ====" + "=" * 10 + " ok"
    out = pruner.clean_repeats(s)
    # 确认长重复被清除（至少 8 连以上）
    assert "-" * 8 not in out
    assert "=" * 8 not in out
    assert "hello" in out and "world" in out and "ok" in out
    import regex  # 注意这里用的是 regex 库，不是内置 re

    latex_special_chars = r'\{}$&%#^_~<>():;@[]-|'
    excluded_chars = regex.escape(latex_special_chars)
    repeat_pat_str = fr"(?:[^\p{{L}}\p{{N}}](?![{excluded_chars}])){{8,}}"
    repeat_pat = regex.compile(repeat_pat_str)

    tests = [
        "正常文字",
        "Hello!!!",                # 3 个 "!"，不足 8 个
        "~~~~~~~~~~",              # 10 个 "~"（LaTeX 特殊符号，应排除，不匹配）
        "!!!!!!!!!!!!",            # 12 个 "!"（不是 LaTeX 符号 → 匹配）
        "----------",              # 10 个 "-"（LaTeX 特殊符号，应排除）
        "????????????????",        # 16 个 "?"（不是 LaTeX 符号 → 匹配）
    ]

    for t in tests:
        m = repeat_pat.search(t)
        print(f"{t:20} -> {'匹配' if m else '不匹配'}")
