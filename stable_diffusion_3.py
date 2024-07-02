import torch
from diffusers import StableDiffusion3Pipeline
import matplotlib.pyplot as plt

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers")
pipe.to("cpu")

image = pipe(
    prompt="a photo of a cat holding a sign that says hello world",
    negative_prompt="",
    num_inference_steps=28,
    height=1024,
    width=1024,
    guidance_scale=7.0,
).images[0]

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.imshow(image)
ax.axis("off")
plt.show()
    
