# prompt_node.py
import json
from typing import List, Dict, Any


class PromptTextEditor:
    """
    ComfyUI节点，用于编辑和处理文本。
    提供一个多行文本输入框和可选的前缀文本，输出编辑后的文本。
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """定义节点的输入参数"""
        return {
            "optional": {
                "append_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "description": "添加文本内容",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("append_text",)
    FUNCTION = "edit_text"
    CATEGORY = "MY_NODES/文本编辑"

    def edit_text(self, append_text: str = "") -> tuple:
        """
        编辑和处理文本的主要方法
        """
        pass
        return (append_text,)


