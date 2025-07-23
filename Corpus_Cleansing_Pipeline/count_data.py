import json
from pathlib import Path

class JsonlCountByConditionsNode:
    """
    Count number of JSONL records in a folder that match multiple field=value conditions.
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
                "conditions": ("STRING", {
                    "multiline": True,
                    "default": "[]",
                    "lazy": True
                }),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("count",)
    OUTPUT_NODE = True
    FUNCTION = "count_matches"
    CATEGORY = "MY_NODES/Tool"

    def count_matches(self, input_folder, conditions):
        input_path = Path(input_folder)
        total_count = 0

        # Parse conditions
        try:
            conditions_list = json.loads(conditions)
            if not isinstance(conditions_list, list):
                raise ValueError
        except Exception as e:
            raise ValueError(f"Invalid conditions format. Please provide JSON list. Error: {e}")

        # Preprocess conditions into list of (field, value)
        condition_pairs = []
        for cond in conditions_list:
            if isinstance(cond, dict) and "field" in cond and "value" in cond:
                condition_pairs.append( (cond["field"], str(cond["value"])) )

        if not condition_pairs:
            raise ValueError("No valid conditions provided. Each condition must have 'field' and 'value'.")

        # Iterate files
        for file in input_path.glob("*.jsonl"):
            with open(file, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if all(str(rec.get(field, "")) == value for field, value in condition_pairs):
                            total_count += 1
                    except json.JSONDecodeError:
                        continue

        return (total_count,)



