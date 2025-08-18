import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
# add_extra_model_paths()



from nodes import NODE_CLASS_MAPPINGS


def main():
    from custom_nodes.Corpus_Cleansing_Pipeline._tests.unzip import import_custom_nodes
    import_custom_nodes()
    with torch.inference_mode():
        # groupexecutorsingle = NODE_CLASS_MAPPINGS["GroupExecutorSingle"]()
        # groupexecutorsingle_2 = groupexecutorsingle.execute_group(
        #     group_name="pdf_pipeline",
        #     repeat_count=1,
        #     delay_seconds=1,
        #     unique_id=1182307044876203691,
        # )

        # prompttexteditor = NODE_CLASS_MAPPINGS["PromptTextEditor"]()
        # prompttexteditor_4 = prompttexteditor.edit_text(append_text="url链接")

        mineru_ocr = NODE_CLASS_MAPPINGS["Mineru_ocr"]()
        from custom_nodes.Corpus_Cleansing_Pipeline._tests.unzip import unzipped_data_homedir
        mineru_ocr_5 = mineru_ocr.mineru(
            read_path=unzipped_data_homedir + "pdf_path",
            out_path=unzipped_data_homedir + "mineru_out_path",
            backend="pipeline",
            type="pdf",
        )

        markdowninfonode = NODE_CLASS_MAPPINGS["MarkdownInfoNode"]()
        multipruner = NODE_CLASS_MAPPINGS["MultiPruner"]()
        batchprocessdocuments_zho = NODE_CLASS_MAPPINGS["BatchProcessDocuments_Zho"]()
        batchgenerateqanode = NODE_CLASS_MAPPINGS["BatchGenerateQANode"]()
        groupexecutorsender = NODE_CLASS_MAPPINGS["GroupExecutorSender"]()

        for q in range(1):
            markdowninfonode_8 = markdowninfonode.execute(
                md_root=get_value_at_index(mineru_ocr_5, 0),
                original_root=unzipped_data_homedir + "pdf_parsed/jsonl",
                output_file=unzipped_data_homedir + "pdf_parsed/jsonl/md_info.jsonl",
                threads=8,
            )

            multipruner_7 = multipruner.run_pruners(
                input_dir=get_value_at_index(markdowninfonode_8, 0),
                output_dir=unzipped_data_homedir + "pdf_parsed/clean",
                num_processes=1,
                min_text_length=100,
                enable_markdown=True,
                enable_email=True,
                enable_link=True,
                enable_ip=True,
                enable_control_char=True,
                enable_repeat_space=True,
                enable_punctuation_clip=True,
                enable_text_circled=True,
                enable_filename=True,
                enable_repeat_char=True,
                enable_html=True,
                enable_answer_sheet_filter=True,
            )

            from os.path import join
            from custom_nodes.Corpus_Cleansing_Pipeline._tests.unzip import comfyui_path
            batchprocessdocuments_zho_6 = batchprocessdocuments_zho.execute_batch_process(
                input_folder=get_value_at_index(multipruner_7, 0),
                output_folder=unzipped_data_homedir + "pdf_parsed/label",
                api_key="None",
                api_url='get_value_at_index(prompttexteditor_4, 0)',
                model_name="qwen",
                valid_files_folder=join(comfyui_path, "custom_nodes/Corpus_Cleansing_Pipeline/config"),
                max_workers=True,
                trigger=True,
            )

            # batchgenerateqanode_1 = batchgenerateqanode.process(
            #     input_folder=get_value_at_index(batchprocessdocuments_zho_6, 0),
            #     output_folder="../example/test_out/pdf/QA",
            #     api_key="None",
            #     api_url=get_value_at_index(prompttexteditor_4, 0),
            #     model_name="qwen",
            #     max_workers=1,
            #     shard_size=100,
            # )
            #
            # groupexecutorsingle_3 = groupexecutorsingle.execute_group(
            #     group_name="QA",
            #     repeat_count=1,
            #     delay_seconds=1,
            #     signal=get_value_at_index(groupexecutorsingle_2, 0),
            #     unique_id=15836557110066989812,
            # )
            #
            # groupexecutorsender_17 = groupexecutorsender.execute(
            #     signal=get_value_at_index(groupexecutorsingle_3, 0),
            #     unique_id=188854885684351832,
            # )


if __name__ == "__main__":
    main()
