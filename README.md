# Set You Straight: Auto-Steering Denoising Trajectories to Sidestep Unwanted Concepts

**Leyang Li, Shilin Lu, Yan Ren, Adams Wai-Kin Kong**

------

## Setup

```bash
conda create -n ant python=3.10
conda activate ant
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install xformers==0.0.24
pip install -r requirements.txt
pip install numpy==1.26.3 
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
```



## Train

### Single Concept

```
cd Single
```

Download the SD1.4 weighs from [here](https://huggingface.co/Haerin1/ANT/blob/main/sd-v1-4-full-ema.ckpt) and move it to `./models/ldm`.

Make sure you have at least 48GB of VRAM on a single GPU, or at least 24GB on each of two GPUs.

```bash
python train-scripts/train_ant_single.py --config configs/ant_gradient.yaml #you can modify parameters in yaml file
```

After this, you can get a folder containing gradient maps under `./gradient`. Use the following command to generate the saliency map by replacing 'prompt_name' with the name gotten from last step (not the path).

```bash
python train-scripts/generate_mask.py --prompt_name 'replace_this'
```

At last, you can use the saliency map to train your model which will be stored in `./models`. Don't forget to modify `mask_path` in yaml file.

```
python train-scripts/train_ant_single.py --config configs/ant_train.yaml
```

### Multiple Concepts

```
cd Multiple
```

Download pre-cached prior knowledge from [here](https://huggingface.co/Haerin1/ANT/tree/main/cache) and move it to ./cache. Then you can run the script to generate LoRA matric for each single concept.

```
pip install diffusers==0.20.0
python train_ant.py --config_file "examples/config.yaml" --erase_type "celebrity"
```

At last, you can fuse the LoRA and get the final model.

```
pip install diffusers==0.22.0
python fuse_lora_ant.py "examples/erase_cele_100.yaml"
```



## Evaluate

Refer to [this page](https://github.com/Shilin-LU/MACE?tab=readme-ov-file#metrics-evaluation) for nudity, celebrity and art style evaluation.

We have uploaded our fine-tuned models to huggingface. If you just want to evaluate the model and do not want to train from the start, you can download these models.

| Erasure Type | Fine-tuned Models                                            |
| ------------ | ------------------------------------------------------------ |
| Nudity       | [huggingface](https://huggingface.co/Haerin1/ANT/blob/main/erase_nudity.pt) |
| Celebrity    | [huggingface](https://huggingface.co/Haerin1/ANT/tree/main/erase_celebrity) |
| Art Style    | [huggingface](https://huggingface.co/Haerin1/ANT/tree/main/erase_art) |



## Acknowledgments

Thanks for the following projects that our code is based on: [ESD](https://github.com/rohitgandikota/erasing), [LECO](https://github.com/p1atdev/LECO), [MACE](https://github.com/Shilin-LU/MACE).
