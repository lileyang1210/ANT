import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, parent_dir)

import argparse
import ast
from pathlib import Path
import gc
import torch
import numpy as np
import random
from tqdm import tqdm
from diffusers import DiffusionPipeline, DDIMScheduler
from utils.lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
from utils import train_util
from utils import model_util
from utils import prompt_util
from utils import config_util
from utils.prompt_util import PromptEmbedsCache, PromptEmbedsPair, PromptSettings
from utils.config_util import RootConfig
import yaml


def seed_everything(TORCH_SEED):
	random.seed(TORCH_SEED)
	os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
	np.random.seed(TORCH_SEED)
	torch.manual_seed(TORCH_SEED)
	torch.cuda.manual_seed_all(TORCH_SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def flush():
    torch.cuda.empty_cache()
    gc.collect()


def train(
    config: RootConfig,
    prompts: list[PromptSettings],
):
    save_path = Path(config.save.path)
#####################################################
    devices = config.train.device
    DEVICE_CUDA = torch.device(devices)
    before_step = config.train.before_step
    alpha_1 = config.train.alpha_1
    alpha_2 = config.train.alpha_2
######################################################
    pipe = DiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipe.safety_checker=None
    pipe.requires_safety_checker=False
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(devices)
    pipe.enable_xformers_memory_efficient_attention()

    modules = DEFAULT_TARGET_REPLACE
    if config.network.type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV


    weight_dtype = config_util.parse_precision(config.train.precision)
    save_weight_dtype = config_util.parse_precision(config.train.precision)

    tokenizer, text_encoder, unet, noise_scheduler = model_util.load_models(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
        v2=config.pretrained_model.v2,
        v_pred=config.pretrained_model.v_pred,
    )

    text_encoder.to(DEVICE_CUDA, dtype=weight_dtype)
    text_encoder.eval()

    unet.to(DEVICE_CUDA, dtype=weight_dtype)
    unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()

    network = LoRANetwork(
        unet,
        rank=config.network.rank,
        multiplier=1.0,
        alpha=config.network.alpha,
        train_method=config.network.training_method,
    ).to(DEVICE_CUDA, dtype=weight_dtype)

    optimizer_module = train_util.get_optimizer(config.train.optimizer)
    optimizer_kwargs = {}
    if config.train.optimizer_args is not None and len(config.train.optimizer_args) > 0:
        for arg in config.train.optimizer_args.split(" "):
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            optimizer_kwargs[key] = value
            
    optimizer = optimizer_module(network.prepare_optimizer_params(), lr=config.train.lr, **optimizer_kwargs)
    lr_scheduler = train_util.get_lr_scheduler(
        config.train.lr_scheduler,
        optimizer,
        max_iterations=config.train.iterations,
        lr_min=config.train.lr / 100,
    )
    criteria = torch.nn.MSELoss()

    print("Prompts")
    for settings in prompts:
        print(settings)


    cache = PromptEmbedsCache()
    prompt_pairs: list[PromptEmbedsPair] = []
    with torch.no_grad():
        for settings in prompts:
            print(settings)
            for prompt in [
                settings.target,
                settings.positive,
                settings.neutral,
                settings.unconditional,
            ]:
                if cache[prompt] == None:
                    cache[prompt] = train_util.encode_prompts(
                        tokenizer, text_encoder, [prompt]
                    )

            prompt_pairs.append(
                PromptEmbedsPair(
                    criteria,
                    cache[settings.target],
                    cache[settings.positive],
                    cache[settings.unconditional],
                    cache[settings.neutral],
                    settings,
                )
            )

    del tokenizer
    del text_encoder

    flush()

    pbar = tqdm(range(config.train.iterations))

    for i in pbar:
        with torch.no_grad():
            noise_scheduler.set_timesteps(
                config.train.max_denoising_steps, device=DEVICE_CUDA
            )

            optimizer.zero_grad()

            prompt_pair: PromptEmbedsPair = prompt_pairs[
                torch.randint(0, len(prompt_pairs), (1,)).item()
            ]
            timesteps_to_plus = torch.randint(
                 0, before_step, (1,)
             )
            timesteps_to_minus = torch.randint(
                 before_step, config.train.max_denoising_steps, (1,)
             )
            

            height, width = (
                prompt_pair.resolution,
                prompt_pair.resolution,
            )
            if prompt_pair.dynamic_resolution:
                height, width = train_util.get_random_resolution_in_bucket(
                    prompt_pair.resolution
                )

            latents = train_util.get_initial_latents(
                noise_scheduler, prompt_pair.batch_size, height, width, 1
            ).to(DEVICE_CUDA, dtype=weight_dtype)

            denoised_latents_plus = train_util.diffusion(
                unet,
                noise_scheduler,
                latents, 
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.target,
                    prompt_pair.batch_size,
                ),
                start_timesteps=0,
                total_timesteps=timesteps_to_plus, 
                guidance_scale=7.5,
            )
            denoised_latents_minus = train_util.diffusion(
                unet,
                noise_scheduler,
                latents,  
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.target,
                    prompt_pair.batch_size,
                ),
                start_timesteps=0,
                total_timesteps=timesteps_to_minus,
                guidance_scale=7.5,
            )
            noise_scheduler.set_timesteps(1000)
            
            current_timestep_plus = noise_scheduler.timesteps[
                int(timesteps_to_plus * 1000 / config.train.max_denoising_steps)
            ]
            current_timestep_minus = noise_scheduler.timesteps[
                int(timesteps_to_minus * 1000 / config.train.max_denoising_steps)
            ]

            positive_latents_plus = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep_plus,
                denoised_latents_plus,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.positive,
                    prompt_pair.batch_size,
                ),
                guidance_scale=7.5,
            ).to("cpu", dtype=torch.float32)
            positive_latents_minus = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep_minus,
                denoised_latents_minus,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.positive,
                    prompt_pair.batch_size,
                ),
                guidance_scale=7.5,
            ).to("cpu", dtype=torch.float32)
            neutral_latents_plus = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep_plus,
                denoised_latents_plus,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.neutral,
                    prompt_pair.batch_size,
                ),
                guidance_scale=0.0,
            ).to("cpu", dtype=torch.float32)
            neutral_latents_minus = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep_minus,
                denoised_latents_minus,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.neutral,
                    prompt_pair.batch_size,
                ),
                guidance_scale=0.0,
            ).to("cpu", dtype=torch.float32)
            unconditional_latents_plus = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep_plus,
                denoised_latents_plus,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.unconditional,
                    prompt_pair.batch_size,
                ),
                guidance_scale=0.0,
            ).to("cpu", dtype=torch.float32)
            unconditional_latents_minus = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep_minus,
                denoised_latents_minus,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.unconditional,
                    prompt_pair.batch_size,
                ),
                guidance_scale=0.0,
            ).to("cpu", dtype=torch.float32)


        with network:
            target_latents_plus = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep_plus,
                denoised_latents_plus,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.target,
                    prompt_pair.batch_size,
                ),
                guidance_scale=7.5,
            ).to("cpu", dtype=torch.float32)
            target_latents_minus = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep_minus,
                denoised_latents_minus,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.target,
                    prompt_pair.batch_size,
                ),
                guidance_scale=7.5,
            ).to("cpu", dtype=torch.float32)
            target_uncondition_latents_plus = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep_plus,
                denoised_latents_plus,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.unconditional,
                    prompt_pair.batch_size,
                ),
                guidance_scale=0.0,
            ).to("cpu", dtype=torch.float32)
            target_uncondition_latents_minus = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep_minus,
                denoised_latents_minus,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.unconditional,
                    prompt_pair.batch_size,
                ),
                guidance_scale=0.0,
            ).to("cpu", dtype=torch.float32)

        positive_latents_plus.requires_grad = False
        positive_latents_minus.requires_grad = False
        neutral_latents_plus.requires_grad = False
        neutral_latents_minus.requires_grad = False
        unconditional_latents_plus.requires_grad = False
        unconditional_latents_minus.requires_grad = False

        prompt_pair.action = 'enhance'
        loss_1 = prompt_pair.loss(
            target_latents=target_latents_plus,
            positive_latents=positive_latents_plus,
            neutral_latents=neutral_latents_plus,
            unconditional_latents=unconditional_latents_plus,
        )
        prompt_pair.action = 'erase'
        loss_2 = prompt_pair.loss(
            target_latents=target_latents_minus,
            positive_latents=positive_latents_minus,
            neutral_latents=neutral_latents_minus,
            unconditional_latents=unconditional_latents_minus,
        )
        loss_1 = loss_1.mean(dim=list(range(1, len(loss_1.shape)))) #* base_weight_plus
        loss_1 = loss_1.mean()
        loss_2 = loss_2.mean(dim=list(range(1, len(loss_2.shape)))) #* base_weight_minus
        loss_2 = loss_2.mean()
        loss_3 = prompt_pair.loss_fn(target_uncondition_latents_plus, unconditional_latents_plus)
        loss_4 = prompt_pair.loss_fn(target_uncondition_latents_minus, unconditional_latents_minus)
        loss_3 = loss_3.mean(dim=list(range(1, len(loss_3.shape)))) #* base_weight_plus
        loss_3 = loss_3.mean()
        loss_4 = loss_4.mean(dim=list(range(1, len(loss_4.shape)))) #* base_weight_minus
        loss_4 = loss_4.mean()
        loss = loss_2 + alpha_1 * loss_4 + alpha_2 * (loss_1 + alpha_1 * loss_3)
        pbar.set_description(f"Loss*1k: {loss.item()*1000:.4f}")

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        del (
            positive_latents_plus,
            positive_latents_minus,
            neutral_latents_plus,
            neutral_latents_minus,
            unconditional_latents_plus,
            unconditional_latents_minus,
            target_latents_plus,
            target_latents_minus,
            latents,
        )
        flush()

    print("Saving...")
    final_path = save_path /f"lr{config.train.lr:.0e}_alpha1_{alpha_1}_alpha2_{alpha_2}_beforeStep{before_step}_iter{config.train.iterations}_rank{config.network.rank}_schedu{config.train.lr_scheduler}/{config.save.name}"
    final_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(
        final_path/ f"{config.save.name}_last.safetensors",
        dtype=save_weight_dtype,
    )

    del (
        unet,
        noise_scheduler,
        loss,
        optimizer,
        network,
    )

    flush()

    print("Done.")

def extract_list(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    yaml_list = [item[0].replace('-', ' ') for item in data['ANT']['multi_concept'][0][0:100]]
    return yaml_list


def main(args):
    config_file = args.config_file
    if args.erase_type == "celebrity":
        yaml_path = 'examples/erase_cele_100.yaml'
    elif args.erase_type == "art":
        yaml_path = 'examples/erase_art_100.yaml'
    else:
        raise ValueError("Invalid erase type. Choose 'celebrity' or 'art'.")
    yaml_list = extract_list(yaml_path)
    seed_everything(42)
    for one_concept in yaml_list:
        config = config_util.load_config_from_yaml(config_file, one_concept)
        prompts = prompt_util.load_prompts_from_yaml(config.prompts_file, one_concept)
        train(config, prompts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        help="Config file for training.",
        default="",
    )
    parser.add_argument(
        "--erase_type",
        help="celebrity or art",
        default="",
    )

    args = parser.parse_args()

    main(args)
