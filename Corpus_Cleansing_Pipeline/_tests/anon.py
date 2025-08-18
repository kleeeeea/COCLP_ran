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
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
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
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS


def main():
    import_custom_nodes()
    with torch.inference_mode():
        prompttexteditor = NODE_CLASS_MAPPINGS["PromptTextEditor"]()
        prompttexteditor_6 = prompttexteditor.edit_text(append_text="模型url链接")

        prompttexteditor_9 = prompttexteditor.edit_text(append_text="模型url链接")

        groupexecutorsingle = NODE_CLASS_MAPPINGS["GroupExecutorSingle"]()
        groupexecutorsingle_10 = groupexecutorsingle.execute_group(
            group_name="个人名字去隐私",
            repeat_count=1,
            delay_seconds=1,
            unique_id=13470163314979802423,
        )

        anonymizechinesenamesnode = NODE_CLASS_MAPPINGS["AnonymizeChineseNamesNode"]()
        anonymizeschoolnamesnode = NODE_CLASS_MAPPINGS["AnonymizeSchoolNamesNode"]()
        groupexecutorsender = NODE_CLASS_MAPPINGS["GroupExecutorSender"]()

        for q in range(1):
            anonymizechinesenamesnode_5 = anonymizechinesenamesnode.process(
                input_folder="inputpath",
                output_folder="outpath",
                api_key="None",
                api_url=get_value_at_index(prompttexteditor_9, 0),
            )

            anonymizeschoolnamesnode_4 = anonymizeschoolnamesnode.process(
                input_folder=get_value_at_index(anonymizechinesenamesnode_5, 0),
                output_folder="outpath",
                api_key="None",
                api_url=get_value_at_index(prompttexteditor_6, 0),
            )

            groupexecutorsingle_7 = groupexecutorsingle.execute_group(
                group_name="学校名字去隐私",
                repeat_count=1,
                delay_seconds=1,
                signal=get_value_at_index(groupexecutorsingle_10, 0),
                unique_id=12343194276760644187,
            )

            groupexecutorsender_8 = groupexecutorsender.execute(
                signal=get_value_at_index(groupexecutorsingle_7, 0),
                unique_id=16148555859788907222,
            )


if __name__ == "__main__":
    main()
