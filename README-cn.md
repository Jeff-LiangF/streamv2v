# StreamV2V

[英文](./README.md) | [中文](./README-cn.md) | [日文](./README-ja.md) 

**[Looking Backward: Streaming Video-to-Video Translation with Feature Banks]()**
<br/>
[梁丰](https://jeff-liangf.github.io/),
[Akio Kodaira](https://scholar.google.co.jp/citations?user=15X3cioAAAAJ&hl=en),
[徐晨丰](https://www.chenfengx.com/),
[Masayoshi Tomizuka](https://me.berkeley.edu/people/masayoshi-tomizuka/),
[Kurt Keutzer](https://people.eecs.berkeley.edu/~keutzer/),
[Diana Marculescu](https://www.ece.utexas.edu/people/faculty/diana-marculescu)
<br/>

[![arXiv](https://img.shields.io/badge/arXiv-2312.17681-b31b1b.svg)](https://arxiv.org/abs/2312.17681)
[![项目页面](https://img.shields.io/badge/Project-Website-orange)](https://jeff-liangf.github.io/projects/streamv2v/)

## 亮点

我们的StreamV2V可以在一块RTX 4090 GPU上实时执行视频到视频的翻译。查看[视频](https://www.youtube.com/watch?v=k-DmQNjXvxA)并[亲自尝试](./demo_w_camera/README.md)！

[![Video](http://img.youtube.com/vi/k-DmQNjXvxA/0.jpg)](https://www.youtube.com/watch?v=k-DmQNjXvxA)

在功能方面，我们的StreamV2V支持面部交换（例如：变成埃隆·马斯克或威尔·史密斯）和视频风格化（例如：变成黏土动画或涂鸦艺术）。查看[视频](https://www.youtube.com/watch?v=N9dx6c8HKBo)并[复现结果](./vid2vid/README.md)！

[![Video](http://img.youtube.com/vi/N9dx6c8HKBo/0.jpg)](https://www.youtube.com/watch?v=N9dx6c8HKBo)

## 安装

请查看[安装指南](./INSTALL.md)。

## 入门

请查看[开始使用说明](./vid2vid/README.md)。

## 本地GPU上的实时摄像头演示

请查看[带摄像头的演示指南](./demo_w_camera/README.md)。

## 许可证

StreamV2V根据[德克萨斯大学奥斯汀分校研究许可证](./LICENSE)进行许可。

## 致谢

StreamV2V在很大程度上依赖于开源社区。我们的代码是从[StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) 和 [LCM-LORA](https://huggingface.co/docs/diffusers/main/zh/using-diffusers/inference_with_lcm_lora) 复制并适应的。除了基础的[SD 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) 模型外，我们还使用了[CIVITAI](https://civitai.com/) 的多种LORAs。

## 引用 StreamV2V :pray:

如果您在研究中使用StreamV2V或希望引用论文中发布的基准结果，请使用以下BibTeX条目。

```BibTeX
StreamV2V TBA

@article{kodaira2023streamdiffusion,
  title={StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation},
  author={Kodaira, Akio and Xu, Chenfeng and Hazama, Toshiki and Yoshimoto, Takanori and Ohno, Kohei and Mitsuhori, Shogo and Sugano, Soichi and Cho, Hanying and Liu, Zhijian and Keutzer, Kurt},
  journal={arXiv preprint arXiv:2312.12491},
  year={2023}
}
