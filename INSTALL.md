## Installation

The installation guide is inherited from [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion?tab=readme-ov-file#installation). If you already have the streamdiffusion env installed, it would also work for streamv2v.

The initial release only tested the Linux env with conda/Docker enviroment.

### Step0: clone this repository

```bash
git clone https://github.com/Jeff-LiangF/streamv2v.git
```

### Step1: Make Environment

You can install StreamV2V via conda, or Docker(explanation below).

```bash
# Using conda (Recommended)
conda create -n streamv2v python=3.10
conda activate streamv2v
```

### Step2: Install PyTorch and other dependencies

Select the appropriate version for your system. Check the [Pytorch doc](https://pytorch.org/).

```bash
# CUDA 11.8
pip3 install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip3 install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu121
```

Install other dependencies

```bash
pip install -r requirements.txt
```

### Step3: Install StreamV2V


#### For Developer (If you want to modify codes)

```bash
python setup.py develop easy_install streamv2v[tensorrt]
python -m streamv2v.tools.install-tensorrt
```

#### For User

```bash
# for Latest Version (recommended)
pip install git+https://github.com/Jeff-LiangF/streamv2v.git@main#egg=streamv2v[tensorrt]
# Install TensorRT extension
python -m streamv2v.tools.install-tensorrt
```

(Only for Windows) You may need to install pywin32.
```bash
pip install --force-reinstall pywin32
```

### Docker Installation (TensorRT Ready)

```bash
git clone https://github.com/Jeff-LiangF/streamv2v.git
cd streamv2v
docker build -t streamv2v:latest -f Dockerfile .
docker run --gpus all -it -v $(pwd):/home/ubuntu/streamv2v streamv2v:latest
```
