# StreamV2V

[English](./README.md) | [中文](./README-cn.md) | [日本語](./README-ja.md)

**[Looking Backward: Streaming Video-to-Video Translation with Feature Banks](https://jeff-liangf.github.io/projects/streamv2v/)**
<br/>
[Feng Liang](https://jeff-liangf.github.io/),
[Akio Kodaira](https://scholar.google.co.jp/citations?user=15X3cioAAAAJ&hl=en),
[Chenfeng Xu](https://www.chenfengx.com/),
[Masayoshi Tomizuka](https://me.berkeley.edu/people/masayoshi-tomizuka/),
[Kurt Keutzer](https://people.eecs.berkeley.edu/~keutzer/),
[Diana Marculescu](https://www.ece.utexas.edu/people/faculty/diana-marculescu)
<br/>

[![arXiv](https://img.shields.io/badge/arXiv-2405.15757-b31b1b.svg)](https://arxiv.org/abs/2405.15757)
[![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://jeff-liangf.github.io/projects/streamv2v/)
[![Huggingface demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/JeffLiang/streamv2v)

## Highlight 

Our StreamV2V could perform real-time video-2-video translation on one RTX 4090 GPU. Check the [video](https://www.youtube.com/watch?v=k-DmQNjXvxA) and [try it by yourself](./demo_w_camera/README.md)!

[![Video](http://img.youtube.com/vi/k-DmQNjXvxA/0.jpg)](https://www.youtube.com/watch?v=k-DmQNjXvxA)

For functionality, our StreamV2V supports face swap (e.g., to Elon Musk or Will Smith) and video stylization (e.g., to Claymation or doodle art). Check the [video](https://www.youtube.com/watch?v=N9dx6c8HKBo) and [reproduce the results](./vid2vid/README.md)!

[![Video](http://img.youtube.com/vi/N9dx6c8HKBo/0.jpg)](https://www.youtube.com/watch?v=N9dx6c8HKBo)

Although StreamV2V is designed for the vid2vid task, it could seamlessly integrate with the txt2img application. Compared with per-image StreamDiffusion, StreamV2V **continuously** generates images from texts, providing a much smoother transition. Check the [video](https://www.youtube.com/watch?v=kFmA0ytcEoA) and [try it by yourself](./demo_continuous_txt2img/README.md)!

[![Video](http://img.youtube.com/vi/kFmA0ytcEoA/0.jpg)](https://www.youtube.com/watch?v=kFmA0ytcEoA)

## Installation

Please see the [installation guide](./INSTALL.md).

## Getting started

Please see [getting started instruction](./vid2vid/README.md).

## Realtime camera demo on GPU

Please see the [demo with camera guide](./demo_w_camera/README.md).

## Continuous txt2img 

Please see the [demo continuous txt2img](./demo_continuous_txt2img/README.md).

## LICENSE

StreamV2V is licensed under a [UT Austin Research LICENSE](./LICENSE).

## Acknowledgements

Our StreamV2V is highly dependended on the open-source community. Our code is copied and adapted from < [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) with [LCM-LORA](https://huggingface.co/docs/diffusers/main/en/using-diffusers/inference_with_lcm_lora). Besides the base [SD 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) model, we also use a variaty of LORAs from [CIVITAI](https://civitai.com/).

## Citing StreamV2V :pray:

If you use StreamV2V in your research or wish to refer to the baseline results published in the paper, please use the following BibTeX entry.

```BibTeX
@article{liang2024looking,
  title={Looking Backward: Streaming Video-to-Video Translation with Feature Banks},
  author={Liang, Feng and Kodaira, Akio and Xu, Chenfeng and Tomizuka, Masayoshi and Keutzer, Kurt and Marculescu, Diana},
  journal={arXiv preprint arXiv:2405.15757},
  year={2024}
}

@article{kodaira2023streamdiffusion,
  title={StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation},
  author={Kodaira, Akio and Xu, Chenfeng and Hazama, Toshiki and Yoshimoto, Takanori and Ohno, Kohei and Mitsuhori, Shogo and Sugano, Soichi and Cho, Hanying and Liu, Zhijian and Keutzer, Kurt},
  journal={arXiv preprint arXiv:2312.12491},
  year={2023}
}
```

## Contributors

<a href="https://github.com/Jeff-LiangF/streamv2v/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Jeff-LiangF/streamv2v" />
</a>