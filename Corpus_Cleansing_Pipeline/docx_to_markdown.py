import os
from markitdown import MarkItDown

class DocxToMarkdown:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_base": ("STRING", {"multiline": False, "default": "docx文件路径"}),
                "output_base": ("STRING", {"multiline": False, "default": "md文件路径"})
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("md_path",)
    FUNCTION = "convert_docx_to_md"
    CATEGORY = "MY_NODES/DocxToMarkdow"
    OUTPUT_NODE = True

    def convert_docx_to_md(self, input_base, output_base):
        # 初始化MarkItDown转换器（禁用插件）
        md = MarkItDown(enable_plugins=False)

        # 遍历输入目录及其所有子目录
        for root, dirs, files in os.walk(input_base):
            for file in files:
                if file.lower().endswith('.docx'):
                    # 获取文件完整路径
                    docx_path = os.path.join(root, file)

                    # 计算相对路径
                    rel_path = os.path.relpath(root, input_base)

                    # 构建输出目录路径（保持相同目录结构）
                    output_dir = os.path.join(output_base, rel_path)

                    # 创建输出目录（如果不存在）
                    os.makedirs(output_dir, exist_ok=True)

                    # 构建输出文件路径（替换扩展名为.md）
                    md_filename = os.path.splitext(file)[0] + '.md'
                    md_path = os.path.join(output_dir, md_filename)

                    try:
                        # 转换文件
                        result = md.convert(docx_path)

                        # 保存转换结果
                        with open(md_path, 'w', encoding='utf-8') as f:
                            f.write(result.text_content)

                        print(f"已转换: {docx_path} -> {md_path}")
                    except Exception as e:
                        print(f"转换失败: {docx_path} - {str(e)}")
        return(output_base,)
    


