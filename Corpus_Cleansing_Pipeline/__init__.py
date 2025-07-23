from .Mineru_node import Mineru_ocr
from .move_md import MOVE_MD
from .auto_label import SampleDataNode, LabelDataNode
from .Copy_Files_by_ID import JsonlCopyFilesByIdNode
from .count_data import JsonlCountByConditionsNode
from .docx_to_markdown import DocxToMarkdown
from .Extract_IDs import JsonlExtractIdsByFieldNode
from .FileClassifier_Node import FileClassifier
from .md_to_jsonl import MarkdownInfoNode
from .multipruner_node_v2 import MultiPruner
from .pipeline_node import BatchProcessDocumentsNode
from .privacy_name import AnonymizeChineseNamesNode
from .privacy_school import AnonymizeSchoolNamesNode
from .QA_extract import BatchGenerateQANode
from .text_node import PromptTextEditor
from .train_label import TrainBertNode, InferenceBertNode
from .unzip_node import UNZIP
from .whisper_novad_node import WhisperNoVADTranscribeNode


NODE_CLASS_MAPPINGS = {
    "Mineru_ocr": Mineru_ocr,
    "MOVE_MD": MOVE_MD,
    "SampleDataNode": SampleDataNode,
    "LabelDataNode": LabelDataNode,
    "JsonlCopyFilesByIdNode": JsonlCopyFilesByIdNode,
    "JsonlCountByConditionsNode": JsonlCountByConditionsNode,
    "DocxToMarkdown": DocxToMarkdown,
    "JsonlExtractIdsByFieldNode": JsonlExtractIdsByFieldNode,
    "FileClassifier": FileClassifier,
    "MarkdownInfoNode": MarkdownInfoNode,
    "MultiPruner": MultiPruner,
    "BatchProcessDocuments_Zho": BatchProcessDocumentsNode,
    "AnonymizeChineseNamesNode": AnonymizeChineseNamesNode,
    "AnonymizeSchoolNamesNode": AnonymizeSchoolNamesNode,
    "BatchGenerateQANode": BatchGenerateQANode,
    "PromptTextEditor": PromptTextEditor,
    "TrainBertNode": TrainBertNode,
    "InferenceBertNode": InferenceBertNode,
    "UNZIP": UNZIP,
    "WhisperNoVADTranscribeNode": WhisperNoVADTranscribeNode
    
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Mineru_ocr": "Mineru_ocr",
    "MOVE_MD": "MD文件移动工具",
    "SampleDataNode": "抽样数据 (蓄水池抽样)",
    "LabelDataNode": "标注数据 (LLM)",
    "JsonlCopyFilesByIdNode": "JSONL Copy Files By ID Node",
    "JsonlCountByConditionsNode": "JSONL Count By Conditions Node",
    "DocxToMarkdown": "DocxToMarkdown",
    "JsonlExtractIdsByFieldNode": "JSONL Extract IDs By Field Node",
    "FileClassifier": "文件分类器",
    "MarkdownInfoNode": "Markdown → JSONL",
    "MultiPruner": "多功能文本清洗器",
    "BatchProcessDocuments_Zho": "文档批量分类处理 (Batch Process)",
    "AnonymizeChineseNamesNode": "Anonymize Chinese Names Node",
    "AnonymizeSchoolNamesNode": "Anonymize School Names Node",
    "BatchGenerateQANode": "Batch Generate QA Node",
    "PromptTextEditor": "文本编辑器",
    "TrainBertNode": "训练BERT分类器",
    "InferenceBertNode": "使用BERT分类器推理",
    "UNZIP": "压缩包解压",
    "WhisperNoVADTranscribeNode": "Whisper 批量转写节点 (含VAD+简体)",
}


