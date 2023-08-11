# This file runs during container build time to get model weights built into the container

# In this example: A Huggingface Stable Diffusion custom model
from diffusers import StableDiffusionXLPipeline
import torch

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    # this should match the model load used in app.py's init function

    from diffusers import StableDiffusionXLPipeline
    import torch

    pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )

if __name__ == "__main__":
    download_model()

