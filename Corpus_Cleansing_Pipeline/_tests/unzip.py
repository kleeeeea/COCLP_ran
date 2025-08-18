import os
import random
import sys
from os.path import join
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




from nodes import NODE_CLASS_MAPPINGS
comfyui_path = '/Users/l/git_repos/ComfyUI'
data_homedir = comfyui_path + '/custom_nodes/Corpus_Cleansing_Pipeline/_tests/'
unzipped_data_homedir = data_homedir + 'unzip/'
def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    sys.path = [
            comfyui_path
               ] + sys.path
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    asyncio.run(init_extra_nodes())
def main():
    import_custom_nodes()
    with torch.inference_mode():

        unzip = NODE_CLASS_MAPPINGS["UNZIP"]()
        unzip_6 = unzip.test(path=join(data_homedir, "raw_input"), max_workers=1, root_path="log_path")

        fileclassifier = NODE_CLASS_MAPPINGS["FileClassifier"]()
        groupexecutorsender = NODE_CLASS_MAPPINGS["GroupExecutorSender"]()

        for q in range(1):
            fileclassifier_1 = fileclassifier.file_out(
                file_path=get_value_at_index(unzip_6, 0),
                vido_path=join(data_homedir, "unzip/video_path"),
                audio_path=join(data_homedir, "unzip/audio_path"),
                image_path=join(data_homedir, "unzip/image_path"),
                pdf_path=join(data_homedir, "unzip/pdf_path"),
                docx_path=join(data_homedir, "unzip/docx_path"),
                overwrite=True,
            )

            groupexecutorsingle_3 = groupexecutorsingle.execute_group(
                group_name="文件分类",
                repeat_count=1,
                delay_seconds=1,
                signal=get_value_at_index(groupexecutorsingle_5, 0),
                unique_id=11249764543010818343,
            )

            groupexecutorsender_4 = groupexecutorsender.execute(
                signal=get_value_at_index(groupexecutorsingle_3, 0),
                unique_id=2133863745919927631,
            )


if __name__ == "__main__":
    main()
