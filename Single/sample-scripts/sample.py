import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, DDIMScheduler
import copy
import csv
import gc
import os
import argparse
from PIL import Image

def flush():
  torch.cuda.empty_cache()
  gc.collect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        required=True,
    )
    parser.add_argument(
        "--csv_path",
        required=True,
    )
    args = parser.parse_args()
    model_path = args.model_path
    csv_path = args.csv_path
    flush()


    pipe = DiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        #custom_pipeline="lpw_stable_diffusion",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    pipe.safety_checker=None
    pipe.requires_safety_checker=False

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to("cuda")
    pipe.unet.load_state_dict(torch.load(model_path))
    lora_unet = copy.deepcopy(pipe.unet)

    width = 512 #@param {type: "number"}
    height = 512 #@param {type: "number"}

    steps = 50  #@param {type:"slider", min:1, max:50, step:1}
    cfg_scale = 7.5 #@param {type:"slider", min:1, max:16, step:0.5}

    with open(csv_path,'r') as file:
        reader = csv.DictReader(file)
        save_path = f'../sample_images'
        os.makedirs(save_path, exist_ok=True)
        for i,row in enumerate(reader):
            seed = row['evaluation_seed']
            prompt = row['prompt']
            pipe.unet = lora_unet
            lora_sample = pipe(
                prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=torch.manual_seed(seed),
            ).images[0]
            lora_sample.save(f'{save_path}/{i}_{seed}.png')

if __name__ == '__main__':
    main()