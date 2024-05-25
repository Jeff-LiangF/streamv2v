## Get started with StreamV2V

[English](./README.md) | [日本語](./README-ja.md)

### Prepartion

We recommend to use [gdown](https://github.com/wkentaro/gdown) to prepare data and models.
```bash
# Install gdown
pip install gdown
pip install --upgrade gdown
```

Download evaluation videos.

```bash
cd vid2vid
gdown https://drive.google.com/drive/folders/1q963FU9I4I8ml9_SeaW4jLb4kY3VkNak -O demo_selfie --folder
```

(Recommended) Download lora weights for better stylization.

```bash
# Make sure you are under the directory of vid2vid
gdown https://drive.google.com/drive/folders/1D7g-dnCQnjjogTPX-B3fttgdrp9nKeKw -O lora_weights --folder
```

| Trigger words                                            | LORA weights     | Source      |
|----------------------------------------------------------|------------------|-------------|
| 'pixelart' ,  'pixel art' ,  'Pixel art' ,  'PixArFK'    | [Google drive](https://drive.google.com/file/d/1_-kEVFw_LnV1J2Nho6nZt4PUbymamypK/view?usp=drive_link) | [Civitai](https://civitai.com/models/185743/8bitdiffuser-64x-or-a-perfect-pixel-art-model) |
| 'lowpoly', 'low poly', 'Low poly'                        | [Google drive](https://drive.google.com/file/d/1ZClfRljzKmxsU1Jj5OMwIuXQcnA1DwO9/view?usp=drive_link) | [Civitai](https://civitai.com/models/110435/y5-low-poly-style) |
| 'Claymation', 'claymation'                               | [Google drive](https://drive.google.com/file/d/1GvPCbrPqJYj0_nRppSc2UD_1eRME-1tG/view?usp=drive_link) | [Civitai](https://civitai.com/models/25258/claymation-miniature) |
| 'crayons', 'Crayons', 'crayons doodle', 'Crayons doodle' | [Google drive](https://drive.google.com/file/d/12ZMOy8CMzwB32RHSmff0h2TJC3lFDBmW/view?usp=drive_link) | [Civitai](https://civitai.com/models/90558/child-type-doodles) |
| 'sketch', 'Sketch', 'pencil drawing', 'Pencil drawing'   | [Google drive](https://drive.google.com/file/d/1NIBujegFMvFdjCW0vdrmD6fbNFKNROE4/view?usp=drive_link) | [Civitai](https://civitai.com/models/155490/pencil-sketch-or) |
| 'oil painting', 'Oil painting'                           | [Google drive](https://drive.google.com/file/d/1fmS3fGeja0RM8YbZtbKw20fjXNzHrnxz/view?usp=drive_link) | [Civitai](https://civitai.com/models/84542/oil-paintingoil-brush-stroke) |

### Evaluation

```bash
# Evaluate a single video
python main.py --input ./demo_selfie/jeff_1.mp4 --prompt "Elon Musk is giving a talk."
python main.py --input ./demo_selfie/jeff_1.mp4 --prompt "Claymation, a man is giving a talk."
```

```bash
# Evaluate a batch of videos
python batch_eval.py --json_file ./demo_selfie/eval_jeff_celebrity.json # Face swap edits
python batch_eval.py --json_file ./demo_selfie/eval_jeff_lorastyle.json # Stylization edits
```

CAUTION: The `--acceleration tensorrt` option is NOT SUPPORTED! I did try to accelerate the model with TensorRT, but due to the dynamic nature of the feature bank, I didn't succeed. If you are an expert on this, please contact me (jeffliang@utexas.edu) and we could discuss how to include you as a contributor. 

### Ablation study using command

```bash
# Do not use feature bank, the model would roll back into per-frame StreamDiffusion
python main.py --input ./demo_selfie/jeff_1.mp4 --prompt "Claymation, a man is giving a talk." --use_cached_attn False --output_dir outputs_streamdiffusion
```

```bash
# Specify the noise strength. Higher the noise_strength means more noise is added to the starting frames.
# Highter strength ususally leads to better edit effects but may sacrifice the consistency. By default, it is 0.4.
python main.py --input ./demo_selfie/jeff_1.mp4 --prompt "Claymation, a man is giving a talk." --noise_strength 0.8 --output_dir outputs_strength
```

```bash
# Specify the diffusion steps. Higher steps ususally lead to higher quality but slower speed.
# By default, it is 4.
python main.py --input ./demo_selfie/jeff_1.mp4 --prompt "Claymation, a man is giving a talk." --diffusion_steps 1 --output_dir outputs_steps
```

### Common Bugs

#### ImportError Issue
- **Error Message**: `ImportError: cannot import name 'packaging' from 'pkg_resources'`.
- **Related GitHub Issue**: [setuptools issue #4961](https://github.com/vllm-project/vllm/issues/4961)

**Potential Workaround**:  
Downgrade the setuptools package to resolve this issue. You can do this by running the following command in your terminal:

```bash
pip install setuptools==69.5.1
```