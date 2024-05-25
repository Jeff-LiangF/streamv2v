# StreamV2V

[英語](./README.md) | [中国語](./README-cn.md) | [日本語](./README-ja.md)

**[Looking Backward: Streaming Video-to-Video Translation with Feature Banks]()**
<br/>
[Feng Liang](https://jeff-liangf.github.io/),
[Akio Kodaira](https://scholar.google.co.jp/citations?user=15X3cioAAAAJ&hl=en),
[Chenfeng Xu](https://www.chenfengx.com/),
[Masayoshi Tomizuka](https://me.berkeley.edu/people/masayoshi-tomizuka/),
[Kurt Keutzer](https://people.eecs.berkeley.edu/~keutzer/),
[Diana Marculescu](https://www.ece.utexas.edu/people/faculty/diana-marculescu)
<br/>

[![arXiv](https://img.shields.io/badge/arXiv-2312.17681-b31b1b.svg)](https://arxiv.org/abs/2312.17681)
[![プロジェクトページ](https://img.shields.io/badge/Project-Website-orange)](https://jeff-liangf.github.io/projects/streamv2v/)

## ハイライト

私たちのStreamV2Vは、4090TI GPUでリアルタイムのビデオ・ツー・ビデオ翻訳を実行することができます。[動画](https://www.youtube.com/watch?v=k-DmQNjXvxA)をチェックして、[自分で試してみてください](./demo_w_camera/README.md)！

[![Video](http://img.youtube.com/vi/k-DmQNjXvxA/0.jpg)](https://www.youtube.com/watch?v=k-DmQNjXvxA)

機能面では、StreamV2Vは顔交換（例：イーロン・マスクやウィル・スミス）やビデオのスタイリゼーション（例：クレイアニメーションやドゥードルアート）をサポートしています。[動画](https://www.youtube.com/watch?v=N9dx6c8HKBo)をチェックして、[結果を再現してみてください](./vid2vid/README.md)！

[![Video](http://img.youtube.com/vi/N9dx6c8HKBo/0.jpg)](https://www.youtube.com/watch?v=N9dx6c8HKBo)

## インストール

[インストールガイド](./INSTALL.md)をご覧ください。

## 入門

[はじめに](./vid2vid/README.md)をご覧ください。

## ローカルGPUでのリアルタイムカメラデモ

[カメラガイド付きデモ](./demo_w_camera/README.md)をご覧ください。

## ライセンス

StreamV2Vは[テキサス大学オースティン校の研究ライセンス](./LICENSE)の下でライセンスされています。

## 謝辞

StreamV2Vはオープンソースコミュニティに大きく依存しています。当社のコードは、[StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) と [LCM-LORA](https://huggingface.co/docs/diffusers/main/en/using-diffusers/inference_with_lcm_lora) からコピーして適応されています。[SD 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) モデルの基本以外にも、[CIVITAI](https://civitai.com/) から多様なLORAsを使用しています。

## StreamV2Vを引用する :pray:

StreamV2Vを研究に使用する場合や、論文で公開されているベースライン結果を参照する場合は、以下のBibTeXエントリを使用してください。

```BibTeX
StreamV2V TBA

@article{kodaira2023streamdiffusion,
  title={StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation},
  author={Kodaira, Akio and Xu, Chenfeng and Hazama, Toshiki and Yoshimoto, Takanori and Ohno, Kohei and Mitsuhori, Shogo and Sugano, Soichi and Cho, Hanying and Liu, Zhijian and Keutzer, Kurt},
  journal={arXiv preprint arXiv:2312.12491},
  year={2023}
}
