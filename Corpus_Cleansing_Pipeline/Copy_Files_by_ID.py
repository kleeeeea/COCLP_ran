import json
import shutil
from pathlib import Path


class JsonlCopyFilesByIdNode:
    """
    Given an ID list, find matching records in JSONL files, extract 'path', and copy files to output_folder.
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
                "id_list": ("LIST",),
                "id_field": ("STRING", {
                    "multiline": False,
                    "default": "id",
                    "lazy": True
                }),
                "path_field": ("STRING", {
                    "multiline": False,
                    "default": "path",
                    "lazy": True
                }),
                "output_folder": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "lazy": True
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("copied_folder",)
    OUTPUT_NODE = True
    FUNCTION = "copy_files"
    CATEGORY = "MY_NODES/Tool"

    def copy_files(self, input_folder, id_list, id_field, path_field, output_folder):
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        id_set = set(str(i) for i in id_list)

        for file in input_path.glob("*.jsonl"):
            with open(file, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        rec_id = str(rec.get(id_field, ""))
                        if rec_id in id_set:
                            src_path = rec.get(path_field, "")
                            if src_path and Path(src_path).is_file():
                                dst_path = output_path / Path(src_path).name
                                shutil.copyfile(src_path, dst_path)
                    except json.JSONDecodeError:
                        continue
                    except Exception:
                        continue

        return (str(output_path),)



