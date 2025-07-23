import os
import shutil
from pathlib import Path
from typing import List


def move_files(file_list: List[str], target_dir: str, root_dir: str, overwrite: bool = False):
    """
    将文件列表中的文件移动到目标文件夹，保持原文件结构

    参数:
        file_list: 源文件路径列表
        target_dir: 目标文件夹路径
        root_dir: 根文件夹路径，用于计算相对路径
        overwrite: 是否覆盖已存在的文件
    """
    # 统计信息
    total_files = len(file_list)
    success_count = 0
    failed_files = []

    # 遍历文件列表并移动
    for i, src_path in enumerate(file_list, 1):
        try:
            # 检查源文件是否存在
            if not os.path.exists(src_path):
                print(f"警告: 文件 '{src_path}' 不存在，跳过")
                failed_files.append(src_path)
                continue

            # 计算相对路径
            relative_path = os.path.relpath(src_path, root_dir)
            dst_path = os.path.join(target_dir, relative_path)

            # 确保目标文件夹存在
            dst_dir = os.path.dirname(dst_path)
            os.makedirs(dst_dir, exist_ok=True)

            # 处理文件已存在的情况
            if os.path.exists(dst_path):
                if overwrite:
                    print(f"覆盖已存在的文件: {dst_path}")
                    # 确保先删除已存在的文件/文件夹
                    if os.path.isdir(dst_path):
                        shutil.rmtree(dst_path)
                    else:
                        os.remove(dst_path)
                else:
                    print(f"警告: 文件 '{dst_path}' 已存在，跳过")
                    failed_files.append(src_path)
                    continue

            # 移动文件
            shutil.copy2(src_path, dst_path)
            success_count += 1

            # 显示进度
            print(f"[{i}/{total_files}] 已移动: {src_path} -> {dst_path}")

        except Exception as e:
            print(f"错误: 移动文件 '{src_path}' 时出错: {str(e)}")
            failed_files.append(src_path)

    # 输出总结
    print("\n==== 操作总结 ====")
    print(f"总共尝试移动: {total_files} 个文件")
    print(f"成功移动: {success_count} 个文件")
    print(f"失败: {len(failed_files)} 个文件")

    if failed_files:
        print("\n失败的文件列表:")
        for file in failed_files:
            print(f"- {file}")
    out_path = target_dir
    return (out_path)


"""媒体文件分类器，用于区分视频、音频、图片和PDF文件"""

# 定义各类文件的后缀集合（统一转为小写处理）
_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm', '.mpeg', '.mpg', '.3gp', '.m4v'}
_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a', '.wma', '.alac', '.ape', '.opus'}
_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.svg', '.gif', '.heic', '.raw'}
_PDF_EXTENSIONS = {'.pdf'}
_docx_EXTENSIONS = {'.docx'}

def classify_files(root_dir: str) -> tuple[List[str], List[str], List[str], List[str]]:
    """
    遍历文件夹及子文件夹，分类所有文件

    参数:
        root_dir: 根文件夹路径

    返回:
        四个列表组成的元组，依次为：
        视频文件路径列表、音频文件路径列表、图片文件路径列表、PDF文件路径列表
    """
    video_list = []
    audio_list = []
    image_list = []
    pdf_list = []
    docx_list=[]

    # 验证文件夹是否存在
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"路径 '{root_dir}' 不是有效的文件夹")

    # 递归遍历所有文件
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            # 获取文件后缀（小写）
            ext = Path(filename).suffix.lower()

            # 根据后缀分类
            if ext in _VIDEO_EXTENSIONS:
                video_list.append(file_path)
            elif ext in _AUDIO_EXTENSIONS:
                audio_list.append(file_path)
            elif ext in _IMAGE_EXTENSIONS:
                image_list.append(file_path)
            elif ext in _PDF_EXTENSIONS:
                pdf_list.append(file_path)
            elif ext in _docx_EXTENSIONS:
                docx_list.append(file_path)

    return video_list, audio_list, image_list, pdf_list, docx_list


class FileClassifier:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                    "lazy": True
                }),
                "vido_path": ("STRING", {
                    "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                    "lazy": True
                }),
                "audio_path": ("STRING", {
                    "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                    "lazy": True
                }),
                "image_path": ("STRING", {
                    "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                    "lazy": True
                }),
                "pdf_path": ("STRING", {
                    "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                    "lazy": True
                }),
                 "docx_path": ("STRING", {
                    "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                    "lazy": True
                }),
                "overwrite": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING","STRING",)
    RETURN_NAMES = ("image_path", "pdf_path", "docx_path", "vido_path", "audio_path",)

    FUNCTION = "file_out"

    OUTPUT_NODE = True

    CATEGORY = "MY_NODES/FileClassifier"

    def file_out(self, file_path, vido_path, audio_path, image_path, pdf_path, docx_path, overwrite):
        video_list, audio_list, image_list, pdf_list, docx_list = classify_files(file_path)
        vido_out = move_files(video_list, vido_path, file_path, overwrite)
        audio_out = move_files(audio_list, audio_path, file_path, overwrite)
        image_out = move_files(image_list, image_path, file_path, overwrite)
        pdf_out = move_files(pdf_list, pdf_path, file_path, overwrite)
        docx_out = move_files(docx_list, docx_path, file_path, overwrite)
        print('**********')
        print(vido_out)
        print(audio_out)
        print(image_out)
        print(pdf_out)
        print(docx_out)
        print('**********')
        return (image_out, pdf_out, docx_out, vido_out, audio_out, )


