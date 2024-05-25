## Installation

The installation guide is inherited from [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion?tab=readme-ov-file#installation). If you already have the streamdiffusion env installed, it would also work for streamv2v.

The initial release only tested the Linux env with conda enviroment.

- [ ] Test pip.
- [ ] Test venv.
- [ ] Test Docker.
- [ ] Test Windows.

### Step0: clone this repository

```bash
git clone 
```

### Step1: Make Environment

You can install StreamV2V via conda, pip, or Docker(explanation below).

```bash
# Using conda (Recommended)
conda create -n streamv2v python=3.10
conda activate streamv2v
```

```bash
# Using virtual environment (venv)
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux
source .venv/bin/activate
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

Install StreamV2V

```bash
#for Latest Version (recommended)
pip install git+https://github.com/Jeff-LiangF/streamv2v.git@main#egg=streamv2v[tensorrt]

#for Stable Version
pip install streamv2v[tensorrt]
```

Install TensorRT extension

```bash
python -m streamv2v.tools.install-tensorrt
```
(Only for Windows) You may need to install pywin32 additionally, if you installed Stable Version(`pip install streamv2v[tensorrt]`).
```bash
pip install --force-reinstall pywin32
```

### Docker Installation (TensorRT Ready)

```bash
git clone https://github.com/cumulo-autumn/StreamDiffusion.git
cd StreamDiffusion
docker build -t stream-diffusion:latest -f Dockerfile .
docker run --gpus all -it -v $(pwd):/home/ubuntu/streamdiffusion stream-diffusion:latest
```
