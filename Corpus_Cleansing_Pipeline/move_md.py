import os
import shutil
import argparse
from pathlib import Path

class MOVE_MD:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "md_source_dir": ("STRING", {"multiline": False, "default": "需要转移的md文件路径"}),
                "file_source_dir": ("STRING", {"multiline": False, "default": "md文件对应的原始文件路径"}),
                "output_dir": ("STRING", {"multiline": False, "default": "转移后路径"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("md_path",)
    FUNCTION = "copy_md_files"
    CATEGORY = "MY_NODES/Tool"
    OUTPUT_NODE = True

    def copy_md_files(self, md_source_dir, pdf_source_dir, output_dir):
        """
        根据PDF源目录的结构，将MD源目录中的.md文件复制到输出目录
        
        参数:
            md_source_dir: 包含.md文件的源目录
            pdf_source_dir: 参考结构的PDF源目录
            output_dir: 输出目录
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 存储PDF源目录中的所有文件及其相对路径
        pdf_files = {}
        
        # 遍历PDF源目录，收集所有文件的相对路径
        for root, _, files in os.walk(pdf_source_dir):
            for file in files:
                # 获取文件的完整路径
                file_path = os.path.join(root, file)
                # 计算文件相对于PDF源目录的相对路径
                rel_path = os.path.relpath(file_path, pdf_source_dir)
                # 获取文件名（不包含扩展名）
                file_name = os.path.splitext(file)[0]
                # 存储文件名及其相对路径（不包含文件名本身）
                pdf_files[file_name] = os.path.dirname(rel_path)
        
        # 遍历MD源目录，复制匹配的.md文件
        copied_count = 0
        not_matched = []
        
        for root, _, files in os.walk(md_source_dir):
            for file in files:
                if file.lower().endswith('.md'):
                    # 获取MD文件名（不包含扩展名）
                    md_file_name = os.path.splitext(file)[0]
                    # 检查是否有匹配的PDF文件
                    if md_file_name in pdf_files:
                        # 获取对应的相对路径
                        rel_path = pdf_files[md_file_name]
                        # 构建目标目录
                        target_dir = os.path.join(output_dir, rel_path)
                        # 确保目标目录存在
                        os.makedirs(target_dir, exist_ok=True)
                        # 构建目标文件路径
                        target_file = os.path.join(target_dir, file)
                        # 源文件路径
                        source_file = os.path.join(root, file)
                        
                        # 复制文件
                        shutil.copy2(source_file, target_file)
                        copied_count += 1
                        print(f"已复制: {source_file} -> {target_file}")
                    else:
                        not_matched.append(file)
        
        # 输出统计信息
        print(f"\n操作完成:")
        print(f"成功复制: {copied_count} 个文件")
        print(f"未匹配的文件: {len(not_matched)} 个")
        
        if not_matched:
            print("\n未匹配的文件列表:")
            for file in not_matched:
                print(f"  - {file}")
        return (output_dir,)
    

