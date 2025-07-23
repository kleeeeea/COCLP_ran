import json
from pathlib import Path

class JsonlExtractIdsByFieldNode:
    """
    Filter JSONL records by field value and collect their IDs into a list.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "lazy": True
                }),
                "field_name": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "lazy": True
                }),
                "field_value": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "lazy": True
                }),
                "id_field": ("STRING", {
                    "multiline": False,
                    "default": "id",
                    "lazy": True
                }),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("id_list",)
    OUTPUT_NODE = True
    FUNCTION = "extract_ids"
    CATEGORY = "MY_NODES/Tool"

    def extract_ids(self, input_folder, field_name, field_value, id_field):
        input_path = Path(input_folder)
        ids = []

        for file in input_path.glob("*.jsonl"):
            with open(file, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if str(rec.get(field_name, "")) == field_value:
                            ids.append(str(rec.get(id_field, "")))
                    except json.JSONDecodeError:
                        continue

        return (ids,)


NODE_CLASS_MAPPINGS = {
    "JsonlExtractIdsByFieldNode": JsonlExtractIdsByFieldNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JsonlExtractIdsByFieldNode": "JSONL Extract IDs By Field Node"
}
