**Read this in other languages: [English](README.md), [中文](README_zh.md).**

# Corpus Cleansing Pipeline: COCLP  

This project provides a series of workflows and tools for data cleaning, developed based on the ComfyUI platform.  

---
## 🧪 Local Deployment Guide  

### 1️⃣ Install ComfyUI  

This project depends on ComfyUI. Please first clone and install it:  

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
```
Please install the relevant dependencies according to the ComfyUI project documentation.   

### 2️⃣ Download Project Source Code
First, clone this project's code repository and initialize submodules:  
```bash
git clone https://github.com/LikeSwim/COCLP.git
cd COCLP/
git submodule update --init --recursive
```

### 3️⃣ Install Dependencies
Enter the project directory and install dependencies:  
```bash
cd COCLP/Corpus_Cleansing_Pipeline/
pip install -r requirements.txt
```

### 🔧 Additional Dependencies
Depends on the following plugins or modules:  
[MinerU](https://github.com/opendatalab/MinerU)  
[rgthree-comfy](https://github.com/rgthree/rgthree-comfy)  
[ComfyUI-to-Python-Extension](https://github.com/pydn/ComfyUI-to-Python-Extension)  
[Comfyui-LG_GroupExecutor](https://github.com/LAOGOU-666/Comfyui-LG_GroupExecutor)  
[faster-whisper](https://github.com/SYSTRAN/faster-whisper)  
Please install the corresponding dependencies according to each project's documentation.  
If you encounter the following error when using faster-whisper:  
```bash
Could not load library libcudnn_ops_infer.so.8
Unable to load any of {libcudnn_cnn.so.9.1.0, libcudnn_cnn.so.9.1, libcudnn_cnn.so.9, libcudnn_cnn.so}
libcudnn_ops_infer.so.8: cannot open shared object file: No such file or directory
```
It indicates that the system lacks the CUDA Deep Neural Network library (cuDNN).  
Install cuDNN (example for apt-based systems):  
```bash
sudo apt update
sudo apt install libcudnn8 libcudnn8-dev -y  
```
Check the local cuDNN corresponding .so files:  
```bash
find / -name "libcudnn_ops.so*" 2>/dev/null
```
Specify the cuDNN path:  
```bash
export LD_LIBRARY_PATH=/path/to/your/cudnn/lib:$LD_LIBRARY_PATH
```

## ▶️ Run the Project
Put the following folders into ComfyUI's custom_nodes folder:  
rgthree-comfy  
ComfyUI-to-Python-Extension  
Comfyui-LG_GroupExecutor  
Corpus_Cleansing_Pipeline  
Then start ComfyUI:  
```bash
cd ComfyUI/
python main.py
```

## 📌 Example Explanation
In the example folder, the following workflow examples are provided:  
Image processing workflow  
PDF document processing workflow  
DOCX document processing workflow  
File decompression and classification workflow  
Data privacy processing workflow  
These workflow files can be imported into ComfyUI and run directly.  

## 💾 Convert Workflow to Python Script
In the ComfyUI interface:  
Click the top-left menu: Workflow ➡ Save as Script  
Save the workflow as an executable .py file  
Run in the terminal:  
```bash
python your_workflow_script.py
```

## 🧩 Acknowledgements
Thanks to the following projects or individuals for their support:  
Thanks to the ComfyUI, MinerU, ComfyUI-to-Python-Extension, Comfyui-LG_GroupExecutor, and faster-whisper teams for providing platform support.