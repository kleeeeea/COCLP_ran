import os
import json
import random
from pathlib import Path
from openai import OpenAI

class AnonymizeChineseNamesNode:
    """
    Anonymize Chinese names in JSONL files in a folder by replacing them with generated fake names.
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
                "output_folder": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "lazy": True
                }),
                "api_key": ("STRING", {"multiline": False}),
                "api_url": ("STRING", {"multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_folder",)
    FUNCTION = "process"
    OUTPUT_NODE = True
    CATEGORY = "MY_NODES/Anonymization"

    def __init__(self):
        pass

    def get_client(self, api_config_str):
        try:
            config = json.loads(api_config_str)
            api_key = config.get("api_key")
            base_url = config.get("base_url")
            return OpenAI(api_key=api_key, base_url=base_url)
        except Exception as e:
            raise ValueError(f"Invalid API config JSON: {e}")

    def generate_fake_chinese_names(self, n=100):
        surnames = [
            "张", "李", "王", "赵", "刘", "陈", "杨", "黄", "吴", "周",
            "徐", "孙", "马", "朱", "胡", "林", "何", "高", "郭", "罗"
        ]
        given_names = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
        fake_names = set()
        while len(fake_names) < n:
            surname = random.choice(surnames)
            given = random.choice(given_names)
            if random.random() < 0.5:
                fake_name = surname + given
            else:
                given2 = random.choice(given_names)
                fake_name = surname + given + given2
            fake_names.add(fake_name)
        return list(fake_names)

    def extract_person_names(self, client, text):
        system_prompt = (
            "你是一个信息抽取助手。"
            "你的任务是从用户给定的文本中，找出所有出现的人名。"
            "如果没有人名，就返回一个空列表。"
            "只返回严格的JSON数组，不要解释或输出其他内容。"
        )
        user_prompt = f"请从以下文本中提取所有人名，返回JSON列表：\n{text}"
        try:
            response = client.chat.completions.create(
                model="qwen",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )
            answer = response.choices[0].message.content
            return json.loads(answer)
        except Exception:
            return []

    def split_text(self, text, chunk_size=4000):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    def replace_names_in_text(self, text, name_mapping):
        for real_name, fake_name in name_mapping.items():
            text = text.replace(real_name, fake_name)
        return text

    def process(self, input_folder, output_folder, api_key, api_url):
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create client with user-specified API config
        api=json.dumps(
                {"api_key": api_key, "base_url": api_url}
            )
        client = self.get_client(api)

        all_real_names = set()
        all_texts = []

        # Step 1: Load and split
        for filename in os.listdir(input_path):
            if filename.endswith(".jsonl"):
                file_path = input_path / filename
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            text = data.get("text", "")
                            chunks = self.split_text(text, 4000)
                            all_texts.append((filename, data, chunks))
                        except Exception as e:
                            print(f"Error reading line in {filename}: {e}")

        # Step 2: Extract names
        print("Step 1: Collecting all person names...")
        for _, _, chunks in all_texts:
            for chunk in chunks:
                names = self.extract_person_names(client, chunk)
                for name in names:
                    all_real_names.add(name)
        print(f"Total unique names found: {len(all_real_names)}")

        # Step 3: Generate fake names
        fake_names = self.generate_fake_chinese_names(max(len(all_real_names), 100))
        name_mapping = {real: fake for real, fake in zip(all_real_names, fake_names)}

        print("Name mapping:")
        for k, v in name_mapping.items():
            print(f"{k} -> {v}")

        # Step 4: Replace and write back
        print("Step 2: Replacing names in texts...")
        outputs = {}
        for filename, data, chunks in all_texts:
            original_text = data.get("text", "")
            replaced_text = self.replace_names_in_text(original_text, name_mapping)
            data["text"] = replaced_text

            if filename not in outputs:
                outputs[filename] = []
            outputs[filename].append(data)

        print("Step 3: Saving processed files...")
        for filename, records in outputs.items():
            out_path = output_path / filename
            with open(out_path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("✅ All done!")

        return (str(output_path),)


