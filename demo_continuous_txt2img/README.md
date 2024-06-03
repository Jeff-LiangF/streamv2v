# Continuous Txt2Img

[![Video](http://img.youtube.com/vi/kFmA0ytcEoA/0.jpg)](https://www.youtube.com/watch?v=kFmA0ytcEoA)

This example is mainly adopted from [StreamDiffusion txt2img](https://github.com/cumulo-autumn/StreamDiffusion/tree/main/demo/realtime-txt2img). 
Although StreamV2V is designed for the vid2vid task, it could seamlessly integrate with the txt2img application. Compared with per-image StreamDiffusion, StreamV2V **continuously** generates images from texts, providing a smooth transition.
The default model is one-step [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo).

Tested with VSCode with remote GPU server. The Live Server extension would help to open the local Chorme page.

## Usage

### install Node.js 18+

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
# Restart terminal
nvm install 18
```

### Install

You need Node.js 18+ and Python 3.10 to run this example.
Please make sure you've installed all dependencies according to the [installation instructions](../../README.md#installation).

```bash
pip install -r requirements.txt
cd frontend
npm i
npm run build
cd ..
python main.py
```

then open `http://0.0.0.0:7861` in your browser. (*If `http://0.0.0.0:7861` does not work well, try `http://localhost:7861`)
