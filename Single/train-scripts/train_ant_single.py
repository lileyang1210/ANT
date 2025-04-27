import sys
sys.path.insert(0, "./")
from omegaconf import OmegaConf
import torch
import os
import yaml
from tqdm import tqdm
import numpy as np
from pathlib import Path

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from diffusers import DiffusionPipeline, DDIMScheduler
import argparse
import random
import argparse
from convertModels import savemodelDiffusers



# Util Functions
def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model_from_config(config, device="cpu"):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model

def seed_everything(TORCH_SEED):
	random.seed(TORCH_SEED)
	os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
	np.random.seed(TORCH_SEED)
	torch.manual_seed(TORCH_SEED)
	torch.cuda.manual_seed_all(TORCH_SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1,t_start=-1,log_every_t=None,till_T=None,verbose=True):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])
    log_t = 100
    if log_every_t is not None:
        log_t = log_every_t
    shape = [4, h // 8, w // 8]
    samples_ddim, inters = sampler.sample(S=ddim_steps,
                                     conditioning=c,
                                     batch_size=n_samples,
                                     shape=shape,
                                     verbose=False,
                                     x_T=start_code,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=uc,
                                     eta=ddim_eta,
                                     verbose_iter = verbose,
                                     t_start=t_start,
                                     log_every_t = log_t,
                                     till_T = till_T
                                    )
    if log_every_t is not None:
        return samples_ddim, inters
    return samples_ddim


def get_models(config_path, devices, last_pt_path):
    model_orig = load_model_from_config(config_path, devices[1])
    sampler_orig = DDIMSampler(model_orig)
    
    model = load_model_from_config(config_path, devices[0])
    if last_pt_path:
        model.load_state_dict(torch.load(last_pt_path))
        model_orig.load_state_dict(torch.load(last_pt_path))

    sampler = DDIMSampler(model)

    return model_orig, sampler_orig, model, sampler

def train_ant(prompt, origin_prompt, seed, train_method, if_gradient, start_guidance, negative_guidance, iterations, lr, before_step, alpha_1, alpha_2, config_path, devices, image_size=512, ddim_steps=50, 
              mask_path=None, last_pt_path=None):

    pipe = DiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipe.safety_checker=None
    pipe.requires_safety_checker=False
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(devices[1])
    
    gradients = {}
    save_path = f'models/{prompt}_method{train_method}_epoch{iterations}_lr{lr}_beforeStep{before_step}_alpha1_{alpha_1}_alpha2_{alpha_2}'

    words = [prompt]
    print(words)
    ddim_eta = 0
    # MODEL TRAINING SETUP

    model_orig, sampler_orig, model, sampler = get_models(config_path, devices, last_pt_path)

    # choose parameters to train based on train_method
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        gradients[name] = 0
        # train all layers except x-attns and time_embed layers
        if train_method == 'noxattn':
            if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
                pass
            else:
                print(name)
                parameters.append(param)
        # train only self attention layers
        if train_method == 'selfattn':
            if 'attn1' in name:
                print(name)
                parameters.append(param)
        # train only x attention layers
        if train_method == 'xattn':
            if 'attn2' in name:
                print(name)
                parameters.append(param)
        # train all layers
        if train_method == 'full':
            print(name)
            parameters.append(param)
        # train all layers except time embed layers
        if train_method == 'notime':
            if not (name.startswith('out.') or 'time_embed' in name):
                print(name)
                parameters.append(param)
        if train_method == 'xlayer':
            if 'attn2' in name:
                if 'output_blocks.6.' in name or 'output_blocks.8.' in name:
                    print(name)
                    parameters.append(param)
        if train_method == 'selflayer':
            if 'attn1' in name:
                if 'input_blocks.4.' in name or 'input_blocks.7.' in name:
                    print(name)
                    parameters.append(param)
    # set model to train
    model.train()
    quick_sample_till_t = lambda x, s, code, t: sample_model(model, sampler,
                                                                 x, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                 start_code=code, till_T=t, verbose=False)

    losses = []
    opt = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()
    history = []

    # TRAINING CODE
    pbar = tqdm(range(iterations))
    for i in pbar:
        word = random.sample(words,1)[0]
        emb_0 = model.get_learned_conditioning([''])
        emb_p = model.get_learned_conditioning([word])
        emb_n = model.get_learned_conditioning([f'{word}'])

        opt.zero_grad()

        t_enc_plus = torch.randint(before_step, (1,), device=devices[0])
        t_enc_minus = torch.randint(before_step, ddim_steps, (1,), device=devices[0])
        # Time step from 1000 to 0
        og_num_plus = round((int(t_enc_plus)/ddim_steps)*1000)
        og_num_minus = round((int(t_enc_minus)/ddim_steps)*1000)
        og_num_lim_plus = round((int(t_enc_plus+1)/ddim_steps)*1000)
        og_num_lim_minus = round((int(t_enc_minus+1)/ddim_steps)*1000)

        t_enc_ddpm_plus = torch.randint(og_num_plus, og_num_lim_plus, (1,), device=devices[0])
        t_enc_ddpm_minus = torch.randint(og_num_minus, og_num_lim_minus, (1,), device=devices[0])
        start_code = torch.randn((1, 4, 64, 64)).to(devices[0])

        with torch.no_grad():
            z_plus = quick_sample_till_t(emb_p.to(devices[0]), start_guidance, start_code, int(t_enc_plus))
            z_minus = quick_sample_till_t(emb_p.to(devices[0]), start_guidance, start_code, int(t_enc_minus))
            e_0_plus = model_orig.apply_model(z_plus.to(devices[1]), t_enc_ddpm_plus.to(devices[1]), emb_0.to(devices[1]))
            e_0_minus = model_orig.apply_model(z_minus.to(devices[1]), t_enc_ddpm_minus.to(devices[1]), emb_0.to(devices[1]))
            e_n0_plus = model.apply_model(z_plus.to(devices[0]), t_enc_ddpm_plus.to(devices[0]), emb_0.to(devices[0]))
            e_n0_minus = model.apply_model(z_minus.to(devices[0]), t_enc_ddpm_minus.to(devices[0]), emb_0.to(devices[0]))
            e_p_plus = model_orig.apply_model(z_plus.to(devices[1]), t_enc_ddpm_plus.to(devices[1]), emb_p.to(devices[1]))
            e_p_minus = model_orig.apply_model(z_minus.to(devices[1]), t_enc_ddpm_minus.to(devices[1]), emb_p.to(devices[1]))
        e_n_plus = model.apply_model(z_plus.to(devices[0]), t_enc_ddpm_plus.to(devices[0]), emb_n.to(devices[0]))
        e_n_minus = model.apply_model(z_minus.to(devices[0]), t_enc_ddpm_minus.to(devices[0]), emb_n.to(devices[0]))
        e_0_plus.requires_grad = False
        e_0_minus.requires_grad = False
        e_p_plus.requires_grad = False
        e_p_minus.requires_grad = False
        # The loss function of ANT model
        loss_1 = criteria(e_n_plus.to(devices[0]), e_0_plus.to(devices[0]) + (negative_guidance*(e_p_plus.to(devices[0]) - e_0_plus.to(devices[0])))) #loss = criteria(e_n, e_0) works the best try 5000 epochs
        loss_3 = criteria(e_n_minus.to(devices[0]), e_0_minus.to(devices[0]) - (negative_guidance*(e_p_minus.to(devices[0]) - e_0_minus.to(devices[0]))))
        loss_2 = criteria(e_0_plus.to(devices[0]), e_n0_plus.to(devices[0]))
        loss_4 = criteria(e_0_minus.to(devices[0]), e_n0_minus.to(devices[0]))
        loss = loss_3 + alpha_2 * loss_4 + alpha_1 * (loss_1 + alpha_2 * loss_2)
        # Update weights to erase the concept
        loss.backward()
        if if_gradient:
            with torch.no_grad():
                for name, param in model.model.diffusion_model.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad.data.cpu()
        if mask_path:
            mask = torch.load(mask_path)
            for name, param in model.model.diffusion_model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name].to(
                    devices[0])

        losses.append(loss.item())
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        opt.step()

        if i%25 == 0 and not if_gradient:
            temp_path = f"{save_path}/temp.pt"
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), temp_path)
            savemodelDiffusers(temp_path, None, None, device='cpu')
            if os.path.exists(temp_path):
                os.remove(temp_path)
    model.eval()

    if if_gradient:
        with torch.no_grad(): 
            for name in gradients:
                gradients[name] = torch.abs_(gradients[name])
            gradient_path = os.path.join(f"gradient/{origin_prompt}_{train_method}_{lr}", f"{prompt}")
            os.makedirs(gradient_path, exist_ok=True)
            torch.save(gradients, os.path.join(gradient_path, f"gradient_{seed}.pt"))
    else:
        final_path = f"{save_path}/final.pt"
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), final_path)
        savemodelDiffusers(final_path, None, None, device='cpu')
        if os.path.exists(final_path):
            os.remove(final_path)

def save_model(model, name, path, num, compvis_config_file=None, diffusers_config_file=None, device='cpu', save_compvis=False, save_diffusers=True):
    # SAVE MODEL
    folder_path = path
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f'{folder_path}/{name}-epoch_{num}.pt'
    else:
        path = f'{folder_path}/{name}.pt'
    if save_compvis:
        torch.save(model.state_dict(), path)
    if save_diffusers:
        print('Saving Model in Diffusers Format')
        savemodelDiffusers(path, compvis_config_file, diffusers_config_file, device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='TrainANT')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()

    config = load_config(args.config)

    prompt = config['prompt']
    train_method = config['train_method']
    if_gradient = config['if_gradient']
    start_guidance = config['start_guidance']
    negative_guidance = config['negative_guidance']
    iterations = config['iterations']
    lr = float(config['lr'])
    before_step = config['before_step']
    alpha_1 = config['alpha1']
    alpha_2 = config['alpha2']
    config_path = config['sd_config_path']
    devices = [f'cuda:{int(d.strip())}' for d in config['devices'].split(',')]
    image_size = config['image_size']
    ddim_steps = config['ddim_steps']
    mask_path = config['mask_path']
    last_pt_path = config['last_pt_path']

    if if_gradient:
        # if you want to train the gradient map, you can set the seed
        seeds = [24, 42, 98, 403, 3407]
        for seed in seeds: 
                if prompt == 'nudity':
                    prompt_list = ['nudity', 'naked', 'erotic', 'sexual']
                    for p in prompt_list:
                            seed_everything(seed)
                            train_ant(prompt=p, origin_prompt=prompt, seed=seed, train_method=train_method, if_gradient=if_gradient, start_guidance=start_guidance, negative_guidance=negative_guidance, iterations=iterations, lr=lr, before_step=before_step, alpha_1=alpha_1, alpha_2=alpha_2, config_path=config_path, devices=devices, image_size=image_size, ddim_steps=ddim_steps, mask_path=mask_path, last_pt_path=last_pt_path)
                else:
                    prompt_list = [prompt]
                    for p in prompt_list:
                        train_ant(prompt=p, origin_prompt=prompt, seed=seed, train_method=train_method, if_gradient=if_gradient, start_guidance=start_guidance, negative_guidance=negative_guidance, iterations=iterations, lr=lr, before_step=before_step, alpha_1=alpha_1, alpha_2=alpha_2, config_path=config_path, devices=devices, image_size=image_size, ddim_steps=ddim_steps, mask_path=mask_path, last_pt_path=last_pt_path)
    else:
        train_ant(prompt=prompt, train_method=train_method, start_guidance=start_guidance, negative_guidance=negative_guidance, iterations=iterations, lr=lr, before_step=before_step, alpha_1=alpha_1, alpha_2=alpha_2, config_path=config_path, devices=devices, image_size=image_size, ddim_steps=ddim_steps, mask_path=mask_path, last_pt_path=last_pt_path)
