import os
import sys
import time
from typing import Literal
import gradio as gr
import torch
from torchvision.io import read_video, write_video
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.wrapper import StreamV2VWrapper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define keyword groups with first keywords as options
style_options = {
    "None": None,  # Add an option for "None" or no selection
    "Pixel art": "pixelart",
    "Low poly": "lowpoly",
    "Claymation": "claymation",
    "Crayons doodle": "crayons",
    "Pencil drawing": "sketch",
    "Oil painting": "oilpainting"
}

def process_video(input_path, prompt, selected_keyword, scale=1.0, guidance_scale=1.0, diffusion_steps=4, noise_strength=0.4, acceleration="xformers", use_denoising_batch=True, use_cached_attn=True, use_feature_injection=True, feature_injection_strength=0.8, feature_similarity_threshold=0.98, cache_interval=4, cache_maxframes=1, use_tome_cache=True, do_add_noise=True, enable_similar_image_filter=False, seed=2):
    output_dir = os.path.join(CURRENT_DIR, "outputs")
    model_id: str = "runwayml/stable-diffusion-v1-5"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    video_info = read_video(input_path)
    video = video_info[0] / 255
    fps = video_info[2]["video_fps"]
    height = int(video.shape[1] * scale)
    width = int(video.shape[2] * scale)

    init_step = int(50 * (1 - noise_strength))
    interval = int(50 * noise_strength) // diffusion_steps
    t_index_list = [init_step + i * interval for i in range(diffusion_steps)]

    stream = StreamV2VWrapper(
        model_id_or_path=model_id,
        mode="img2img",
        t_index_list=t_index_list,
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        do_add_noise=do_add_noise,
        output_type="pt",
        enable_similar_image_filter=enable_similar_image_filter,
        similar_image_filter_threshold=0.98,
        use_denoising_batch=use_denoising_batch,
        use_cached_attn=use_cached_attn,
        use_feature_injection=use_feature_injection,
        feature_injection_strength=feature_injection_strength,
        feature_similarity_threshold=feature_similarity_threshold,
        cache_interval=cache_interval,
        cache_maxframes=cache_maxframes,
        use_tome_cache=use_tome_cache,
        seed=seed,
    )
    
    if selected_keyword and selected_keyword != "None":
        prompt = style_options[selected_keyword] + "," + prompt
        
    stream.prepare(
        prompt= prompt,
        num_inference_steps=50,
        guidance_scale=guidance_scale,
    )

    # Load the corresponding LORA based on the selected keyword
    if selected_keyword and selected_keyword != "None":
        print("&&&&&&&&&:",style_options[selected_keyword])
        if style_options[selected_keyword] == "pixelart":
            stream.stream.load_lora("./lora_weights/PixelArtRedmond15V-PixelArt-PIXARFK.safetensors", adapter_name='pixelart')
            stream.stream.pipe.set_adapters(["lcm", "pixelart"], adapter_weights=[1.0, 1.0])
        elif style_options[selected_keyword] == "lowpoly":
            stream.stream.load_lora("./lora_weights/low_poly.safetensors", adapter_name='lowpoly')
            stream.stream.pipe.set_adapters(["lcm", "lowpoly"], adapter_weights=[1.0, 1.0])
        elif style_options[selected_keyword] == "claymation":
            print("ssssssssssssssssssssssssssssssssssssssssssssssssssss")
            stream.stream.load_lora("./lora_weights/Claymation.safetensors", adapter_name='claymation')
            stream.stream.pipe.set_adapters(["lcm", "claymation"], adapter_weights=[1.0, 1.0])
        elif style_options[selected_keyword] == "crayons":
            stream.stream.load_lora("./lora_weights/doodle.safetensors", adapter_name='crayons')
            stream.stream.pipe.set_adapters(["lcm", "crayons"], adapter_weights=[1.0, 1.0])
        elif style_options[selected_keyword] == "sketch":
            stream.stream.load_lora("./lora_weights/Sketch_offcolor.safetensors", adapter_name='sketch')
            stream.stream.pipe.set_adapters(["lcm", "sketch"], adapter_weights=[1.0, 1.0])
        elif style_options[selected_keyword] == "oilpainting":
            stream.stream.load_lora("./lora_weights/bichu-v0612.safetensors", adapter_name='oilpainting')
            stream.stream.pipe.set_adapters(["lcm", "oilpainting"], adapter_weights=[1.0, 1.0])

    video_result = torch.zeros(video.shape[0], height, width, 3)

    for _ in range(stream.batch_size):
        stream(image=video[0].permute(2, 0, 1))

    inference_time = []
    for i in tqdm(range(video.shape[0])):
        iteration_start_time = time.time()
        output_image = stream(video[i].permute(2, 0, 1))
        video_result[i] = output_image.permute(1, 2, 0)
        iteration_end_time = time.time()
        inference_time.append(iteration_end_time - iteration_start_time)

    print(f'Avg time: {sum(inference_time[20:]) / len(inference_time[20:])}')

    video_result = (video_result * 255).byte()
    prompt_txt = prompt.replace(' ', '-')
    input_vid = input_path.split('/')[-1]
    output = os.path.join(output_dir, f"{input_vid.rsplit('.', 1)[0]}_{prompt_txt}.mp4")
    write_video(output, video_result, fps=fps, options={"crf": "20"})
    return output

iface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Video(label="Input Video"),
        gr.Textbox(label="Prompt"),
        gr.Radio(list(style_options.keys()), label="Select Style"),
        gr.Slider(minimum=0.1, maximum=2.0, value=1.0, label="Scale"),
        gr.Slider(minimum=0.1, maximum=2.0, value=1.0, label="Guidance Scale"),
        gr.Slider(minimum=1, maximum=10, value=4, step=1, label="Diffusion Steps"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.4, label="Noise Strength"),
        gr.Radio(["none", "xformers"], value="xformers", label="Acceleration"),
        #gr.Radio(["none", "xformers", "tensorrt"], value="xformers", label="Acceleration"),
        gr.Checkbox(value=True, label="Use Denoising Batch"),
        gr.Checkbox(value=True, label="Use Cached Attention"),
        gr.Checkbox(value=True, label="Use Feature Injection"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.8, label="Feature Injection Strength"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.98, label="Feature Similarity Threshold"),
        gr.Slider(minimum=1, maximum=10, value=4, step=1, label="Cache Interval"),
        gr.Slider(minimum=1, maximum=10, value=1, step=1, label="Cache Max Frames (If want to use > 1, set Token Merging as False)"),
        gr.Checkbox(value=True, label="Use Token Merging Cache"),
        gr.Checkbox(value=True, label="Do Add Noise"),
        gr.Checkbox(value=False, label="Enable Similar Image Filter"),
        gr.Slider(minimum=0, maximum=10, value=2, step=1, label="Seed"),
    ],
    outputs="video",
    title="Video Translation",
    description="Perform video-to-video translation with StreamV2V.",
    allow_flagging=False
)

if __name__ == "__main__":
    iface.launch()