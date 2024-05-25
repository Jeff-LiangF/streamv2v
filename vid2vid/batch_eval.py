import os
import json
import argparse
import subprocess

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process a JSON file contains multiple edits.")
    parser.add_argument('--json_file', type=str, help='The path to the JSON file to process.')
    return parser.parse_args()

data = []
args = parse_arguments()
json_file = args.json_file

# Load the JSON data
with open(json_file, "r") as file:
    for line in file:
        data.append(json.loads(line))

for item in data:
    file_path = item["file_path"]
    src_vid_name = item["src_vid_name"]
    prompt = item["prompt"]
    diffusion_steps = item["diffusion_steps"]
    noise_strength = item["noise_strength"]
    try:
        model_id = item["model_id"]
    except:
        model_id = "runwayml/stable-diffusion-v1-5"
    command = [
        'python', "main.py",
        "--input", f"{file_path}/{src_vid_name}.mp4",
        "--prompt", prompt,
        "--model_id", model_id,
        "--diffusion_steps", diffusion_steps,
        "--noise_strength", noise_strength,
        "--acceleration", "xformers",
        "--use_cached_attn",
        "--use_feature_injection",
        "--cache_maxframes", "1",
        "--use_tome_cache",
        "--do_add_noise", 
        "--guidance_scale", "1.0" 
    ]
    subprocess.run(command)
