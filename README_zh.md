**其他语言版本: [English](README.md), [中文](README_zh.md).**

# 语料清洗管线 Corpus Cleansing Pipeline： COCLP

本项目提供了一系列用于数据清理的工作流和工具，基于 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 平台开发。

---

## 🧪 本地部署指南

### 1️⃣ 安装 ComfyUI

本项目依赖 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)，请先克隆并安装：
```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
```
请根据 ComfyUI 项目文档安装相关依赖。

### 2️⃣ 下载项目源码
首先克隆本项目的代码仓库，并初始化子模块：
```bash
git clone https://github.com/LikeSwim/COCLP.git
cd COCLP/
git submodule update --init --recursive
```
### 3️⃣ 安装依赖
进入项目目录并安装依赖：
```bash
cd COCLP/Corpus_Cleansing_Pipeline/
pip install -r requirements.txt
```
### 🔧 额外依赖
依赖以下插件或模块：  
[MinerU](https://github.com/opendatalab/MinerU)  
[rgthree-comfy](https://github.com/rgthree/rgthree-comfy)  
[ComfyUI-to-Python-Extension](https://github.com/pydn/ComfyUI-to-Python-Extension)  
[Comfyui-LG_GroupExecutor](https://github.com/LAOGOU-666/Comfyui-LG_GroupExecutor)  
[faster-whisper](https://github.com/SYSTRAN/faster-whisper)  
请根据各项目说明文档安装对应依赖。   
使用faster-whisper如遇到以下错误:  
```bash
Could not load library libcudnn_ops_infer.so.8
Unable to load any of {libcudnn_cnn.so.9.1.0, libcudnn_cnn.so.9.1, libcudnn_cnn.so.9, libcudnn_cnn.so}
libcudnn_ops_infer.so.8: cannot open shared object file: No such file or directory
```
说明系统缺少 CUDA 深度神经网络库 （cuDNN）。  
安装 cuDNN（基于 apt 的系统的示例）：  
```bash
sudo apt update
sudo apt install libcudnn8 libcudnn8-dev -y  
```
检查本地cuDNN对应的.so文件
```bash
find / -name "libcudnn_ops.so*" 2>/dev/null
```
指定 cuDNN 路径
```bash
export LD_LIBRARY_PATH=/path/to/your/cudnn/lib:$LD_LIBRARY_PATH
```

## ▶️ 运行项目
将以下文件夹放入 ComfyUI 的 custom_nodes 文件夹中：  
rgthree-comfy  
ComfyUI-to-Python-Extension  
Comfyui-LG_GroupExecutor  
Corpus_Cleansing_Pipeline  
然后启动 ComfyUI：
```bash
cd ComfyUI/
python main.py
```

## 📌 示例说明
在 example 文件夹中，提供了以下工作流示例：  
图像处理工作流  
PDF文档处理工作流  
DOCX文档处理工作流  
解压文件并分类工作流  
数据隐私处理工作流  
可以将这些工作流文件导入 ComfyUI 中直接运行。

## 💾 将工作流转为 Python 脚本
在 ComfyUI 界面中：
点击左上角菜单：Workflow ➡ Save as Script
将工作流保存为可执行的 .py 文件
在终端中运行：
```bash
python your_workflow_script.py
```

## 🧩 致谢  
感谢以下项目或人员的支持：  
感谢 ComfyUI, MinerU, ComfyUI-to-Python-Extension, Comfyui-LG_GroupExecutor, faster-whisper团队提供的平台支持。