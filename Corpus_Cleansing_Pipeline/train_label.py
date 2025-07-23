import os
import json
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

# 确保 transformers 和 datasets 库已安装
try:
    from datasets import Dataset, DatasetDict, load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )
except ImportError:
    print("请安装必要的库: pip install transformers datasets scikit-learn torch")
    # 退出或适当地处理依赖缺失问题
    exit()


# ====================================================================================================
# Functions from train.py (源于 train.py 的函数)
# ====================================================================================================

def run_full_training_pipeline(
        base_model_path: str,  # 预训练模型的文件路径
        input_data_jsonl_path: str,  # 原始标注数据文件 (JSONL格式) 的路径
        output_model_save_dir: str = None,  # 模型保存目录，None则自动生成带时间戳的目录
        learning_rate: float = 1e-5,  # 模型的学习率
        train_batch_size: int = 8,  # 训练阶段的批次大小
        eval_batch_size: int = 64,  # 评估阶段的批次大小
        num_train_epochs: int = 5,  # 训练的总轮次
        label_smoothing_factor: float = 0.1,  # 标签平滑的因子
        test_split_ratio: float = 0.15,  # 用于测试集的数据比例
        validation_split_ratio: float = 0.15,  # 用于验证集的数据比例 (从剩余数据中划分)
        random_seed: int = 42,  # 用于数据切分和模型初始化的随机种子
        cuda_device_id: str = "0",  # 指定用于训练的CUDA设备ID (例如 "0", "1")
        max_sequence_length: int = 512,  # 输入序列的最大长度
        early_stopping_patience: int = 2,  # 早停机制的耐心值
):
    """
    BERT模型训练的主流程。
    """
    print("--- 开始完整的模型训练流程 ---")

    # 如果未指定，根据当前时间生成模型保存目录名称
    if output_model_save_dir is None:
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_model_save_dir = f"output_model/{current_time_str}-finetuned_model"

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device_id  # 设置CUDA设备
    os.makedirs(output_model_save_dir, exist_ok=True)  # 创建模型保存目录

    # 设置随机种子以保证结果可复现
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    print(f"模型将保存到: {output_model_save_dir}")
    print(f"CUDA设备设置为: {cuda_device_id}")

    # 自定义 Trainer (带标签平滑)
    class CustomTrainer(Trainer):
        def __init__(self, *args, label_smoothing_factor, **kwargs):
            super().__init__(*args, **kwargs)
            # 初始化交叉熵损失函数，并应用标签平滑
            self.loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing_factor)

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss = self.loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # 数据准备：加载、预处理和切分数据集
    print(f"正在加载数据: {input_data_jsonl_path}...")
    try:
        raw_dataset = load_dataset("json", data_files=input_data_jsonl_path)["train"]
    except FileNotFoundError:
        print(f"错误: 输入数据文件未找到于 {input_data_jsonl_path}")
        return

    def preprocess_item(example):
        question = example["qa_pair"][0]["content"]
        answer = example["qa_pair"][1]["content"]
        combined_text = f"Question: {question}\nAnswer: {answer}"  # 组合问答对
        return {"line": combined_text, "label": int(example["label"])}

    processed_dataset = raw_dataset.map(preprocess_item, remove_columns=list(raw_dataset.features))

    # 第一次切分：分离测试集
    train_val_test_split = processed_dataset.train_test_split(test_size=test_split_ratio, seed=random_seed)
    train_val_dataset = train_val_test_split["train"]
    test_dataset = train_val_test_split["test"]

    # 第二次切分：从剩余的train_val_dataset中分离验证集
    val_ratio_relative_to_train_val = validation_split_ratio / (1 - test_split_ratio)
    if val_ratio_relative_to_train_val >= 1.0:
        raise ValueError("验证集比例过大。")

    train_val_split = train_val_dataset.train_test_split(test_size=val_ratio_relative_to_train_val, seed=random_seed)
    train_dataset = train_val_split["train"]
    validation_dataset = train_val_split["test"]

    dataset_dict = DatasetDict({"train": train_dataset, "validation": validation_dataset, "test": test_dataset})
    print(
        f"数据集切分完成: 训练集={len(dataset_dict['train'])}, 验证集={len(dataset_dict['validation'])}, 测试集={len(dataset_dict['test'])}")

    # 分词并设置PyTorch格式
    print("正在对数据集进行分词...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    def _tokenize_function(batch):
        # 使用tokenizer进行分词、填充和截断
        return tokenizer(batch["line"], padding="max_length", truncation=True, max_length=max_sequence_length)

    tokenized_datasets = dataset_dict.map(_tokenize_function, batched=True)
    # 为PyTorch设置列格式
    for split in tokenized_datasets:
        tokenized_datasets[split].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    print("分词和格式设置完成。")

    # 模型初始化
    unique_labels = np.unique(tokenized_datasets["train"]["label"].cpu().numpy())
    num_labels = len(unique_labels)
    print(f"模型将使用 {num_labels} 个类别进行初始化。")

    model = AutoModelForSequenceClassification.from_pretrained(base_model_path, num_labels=num_labels)

    # 训练参数设置
    training_args = TrainingArguments(
        output_dir=output_model_save_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="epoch",  # 每个epoch结束时评估
        save_strategy="epoch",  # 每个epoch结束时保存模型
        save_total_limit=1,  # 只保存最佳模型
        logging_dir=os.path.join(output_model_save_dir, "logs"),  # 日志目录
        load_best_model_at_end=True,  # 训练结束后加载最佳模型
        metric_for_best_model="eval_loss",  # 根据验证集损失选择最佳模型
        bf16=torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8,  # 如果GPU支持，使用bf16精度
        tf32=True,  # 启用TF32
        seed=random_seed,  # 随机种子
        report_to="none"  # 不向外部报告
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        label_smoothing_factor=label_smoothing_factor,
    )

    print('开始模型训练...')
    trainer.train()
    print('模型训练完成。')

    final_model_path = os.path.join(output_model_save_dir, "best_model")
    trainer.save_model(final_model_path)
    print(f'最佳训练模型已保存到: {final_model_path}')

    # --- 测试集评估 ---
    print("\n正在测试集上进行详细评估...")
    predictions_output = trainer.predict(tokenized_datasets["test"])
    predictions = np.argmax(predictions_output.predictions, axis=1)  # 预测类别
    true_labels = predictions_output.label_ids  # 真实标签

    # 计算评估指标
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')  # 考虑类别不平衡，使用加权F1
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    cm = confusion_matrix(true_labels, predictions)

    print(f"测试集准确率 (Accuracy): {accuracy:.4f}")
    print(f"测试集F1分数 (F1 Score, Weighted): {f1:.4f}")
    print(f"测试集精确率 (Precision, Weighted): {precision:.4f}")
    print(f"测试集召回率 (Recall, Weighted): {recall:.4f}")
    print("\n混淆矩阵 (Confusion Matrix):\n", cm)
    print("--- 完整的模型训练流程结束 ---")


# ====================================================================================================
# Functions from inference.py (源于 inference.py 的函数)
# ====================================================================================================

def inference_pipeline(
        model_path: str,  # 微调后的模型路径
        input_data_base_dir: str,  # 待处理数据的根目录
        output_acc_base_dir: str,  # 结果输出的根目录
        max_sequence_length: int = 512,  # 分词器的最大序列长度
        inference_batch_size: int = 64,  # 推理时的批次大小 (每个GPU的批次大小)
        positive_threshold: float = 0.5,  # 判断为正例的概率阈值
        cuda_device_ids: list = None,  # GPU ID 列表，例如 [0, 1, 6]
):
    """
    对指定目录下的所有.jsonl文件进行递归推理，只保留正例，并按原目录结构输出。
    """
    print("--- 开始大规模模型推理流程 ---")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # 设置CUDA_VISIBLE_DEVICES环境变量
    if cuda_device_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cuda_device_ids))
        print(f"CUDA设备设置为: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        print("未指定CUDA设备ID，Transformer将自动选择可用设备。")

    if not torch.cuda.is_available():
        print("警告: CUDA设备不可用，将使用CPU进行推理。")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"错误: 无法加载模型或分词器于路径 {model_path}。错误: {e}")
        return

    # 定义数据预处理和分词函数
    def preprocess_item_for_inference(example):
        if example.get("qa_pair") and isinstance(example["qa_pair"], list) and len(example["qa_pair"]) >= 2:
            user_content = example["qa_pair"][0].get("content", "")
            assistant_content = example["qa_pair"][1].get("content", "")
            combined_text = f"Question: {user_content}\nAnswer: {assistant_content}"
        else:
            combined_text = ""
        return {"line": combined_text, "original_json_data": example}

    def _tokenize_function(batch):
        return tokenizer(batch["line"], padding="max_length", truncation=True, max_length=max_sequence_length)

    # 设置 Trainer 用于推理
    inference_args = TrainingArguments(
        output_dir="./tmp_inference_output",  # 临时输出目录
        per_device_eval_batch_size=inference_batch_size,
        dataloader_num_workers=os.cpu_count() // 2 or 1,
        report_to="none",
        no_cuda=not torch.cuda.is_available()
    )
    trainer = Trainer(model=model, args=inference_args, tokenizer=tokenizer)

    # 初始化总计数器
    total_original_items_overall = 0
    total_retained_items_overall = 0

    # 递归遍历目录并处理.jsonl文件
    for root, _, files in os.walk(input_data_base_dir):
        for file in files:
            if file.endswith(".jsonl"):
                input_file_path = os.path.join(root, file)
                # 构建输出文件路径，保持目录结构
                relative_path = os.path.relpath(input_file_path, input_data_base_dir)
                output_file_dir = os.path.join(output_acc_base_dir, os.path.dirname(relative_path))
                os.makedirs(output_file_dir, exist_ok=True)
                output_file_path = os.path.join(output_file_dir, file)

                print(f"\n正在处理文件: {input_file_path}")
                try:
                    # 加载原始数据
                    raw_inference_dataset = load_dataset("json", data_files=input_file_path)["train"]
                    total_items_in_file = len(raw_inference_dataset)
                    total_original_items_overall += total_items_in_file  # 累加到总计数器

                    # 预处理、分词和推理
                    processed_inference_dataset = raw_inference_dataset.map(preprocess_item_for_inference,
                                                                            remove_columns=raw_inference_dataset.column_names)
                    processed_inference_dataset = processed_inference_dataset.filter(lambda x: x.get("line", "") != "")
                    tokenized_inference_dataset = processed_inference_dataset.map(_tokenize_function, batched=True)
                    tokenized_inference_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

                    predictions_output = trainer.predict(tokenized_inference_dataset)
                    logits = predictions_output.predictions
                    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
                    positive_probabilities = probabilities[:, 1]  # 假设标签1是正例的索引

                    # 筛选并保存正例
                    positive_predictions_data = []
                    for i, processed_item in enumerate(processed_inference_dataset):
                        if positive_probabilities[i] >= positive_threshold:
                            positive_predictions_data.append(processed_item["original_json_data"])

                    with open(output_file_path, 'w', encoding='utf-8') as f:
                        for item in positive_predictions_data:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')

                    num_retained_items_in_file = len(positive_predictions_data)
                    total_retained_items_overall += num_retained_items_in_file  # 累加到总计数器
                    print(f"文件 '{input_file_path}' 处理完成。保留了 {num_retained_items_in_file} 条正例。")
                except Exception as e:
                    print(f"处理文件 '{input_file_path}' 时发生错误: {e}")
                    continue  # 继续处理下一个文件

    # 所有文件处理完毕后，计算并打印总体保留率
    overall_retention_rate = (
                                         total_retained_items_overall / total_original_items_overall) * 100 if total_original_items_overall > 0 else 0
    print("\n--- 大规模模型推理流程完成 ---")
    print(f"总体原始条目数: {total_original_items_overall}")
    print(f"总体保留正例数: {total_retained_items_overall}")
    print(f"总体清洗保留率: {overall_retention_rate:.2f}%")


# ====================================================================================================
# ComfyUI Node Definitions (ComfyUI 节点定义)
# ====================================================================================================

class TrainBertNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model_path": ("STRING", {"default": "./hf_models/deberta-v3-base"}),
                "input_data_jsonl_path": ("STRING", {"default": "./labeled_output.jsonl"}),
                "output_model_save_dir": ("STRING", {"default": "./output_model"}),
                "learning_rate": ("FLOAT", {"default": 1e-5, "step": 1e-6, "display": "number"}),
                "train_batch_size": ("INT", {"default": 8, "min": 1}),
                "eval_batch_size": ("INT", {"default": 64, "min": 1}),
                "num_train_epochs": ("INT", {"default": 5, "min": 1}),
                "max_sequence_length": ("INT", {"default": 512, "min": 64, "step": 64}),
                "cuda_device_id": ("STRING", {"default": "0"}),
            },
            "optional": {
                "label_smoothing_factor": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "test_split_ratio": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.99, "step": 0.01}),
                "validation_split_ratio": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.99, "step": 0.01}),
                "random_seed": ("INT", {"default": 42}),
                "early_stopping_patience": ("INT", {"default": 2, "min": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "train_model"
    CATEGORY = "MY_NODES/BERT"

    def train_model(self, **kwargs):
        run_full_training_pipeline(**kwargs)
        return (f"训练完成。模型保存在 {kwargs.get('output_model_save_dir')}",)


class InferenceBertNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "./output_model/best_model"}),
                "input_data_base_dir": ("STRING", {"default": "./source_data"}),
                "output_acc_base_dir": ("STRING", {"default": "./filtered_data"}),
                "inference_batch_size": ("INT", {"default": 64, "min": 1}),
                "positive_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_sequence_length": ("INT", {"default": 512, "min": 64, "step": 64}),
                "cuda_device_ids": ("STRING", {"default": "0", "placeholder": "例如: 0,1,2"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference_data"
    CATEGORY = "MY_NODES/BERT"

    def inference_data(self, model_path, input_data_base_dir, output_acc_base_dir, inference_batch_size,
                       positive_threshold, max_sequence_length, cuda_device_ids):
        # 将逗号分隔的GPU ID字符串解析为整数列表
        parsed_cuda_ids = None
        if cuda_device_ids and cuda_device_ids.strip():
            try:
                parsed_cuda_ids = [int(i.strip()) for i in cuda_device_ids.split(',')]
            except ValueError:
                return (f"错误: 无效的 'cuda_device_ids'。请输入一个逗号分隔的整数列表 (例如: '0,1')。",)

        inference_pipeline(
            model_path=model_path,
            input_data_base_dir=input_data_base_dir,
            output_acc_base_dir=output_acc_base_dir,
            max_sequence_length=max_sequence_length,
            inference_batch_size=inference_batch_size,
            positive_threshold=positive_threshold,
            cuda_device_ids=parsed_cuda_ids,
        )
        return (f"对 {input_data_base_dir} 中的数据推理完成。结果已保存到 {output_acc_base_dir}",)


# ====================================================================================================
# Node Mappings for ComfyUI (ComfyUI 节点映射)
# ====================================================================================================

