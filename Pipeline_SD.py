from diffusers import DDPMPipeline
import torch
from PIL import Image
print(torch.cuda.is_available())
ddpm = DDPMPipeline.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to("cuda")
image = ddpm(num_inference_steps=25).images[0]
image.save("./image.png")