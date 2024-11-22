import torch
from diffusers import FluxPipeline
from datetime import datetime
import os
import time

class ImageGenerator:

    def __init__(self, model_name="black-forest-labs/FLUX.1-schnell", output_dir="output"):
        self.model_name = model_name
        self.output_dir = output_dir
        self._initialize_pipeline()
        os.makedirs(output_dir, exist_ok=True)


    def _initialize_pipeline(self):
        print(f"Loading model: {self.model_name}")
        self.pipe = FluxPipeline.from_pretrained(self.model_name, torch_dtype=torch.bfloat16)
        self.pipe.enable_sequential_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
                

    def generate_images(self, prompt, num_images, height=768, width=1360, num_inference_steps=4, max_sequence_length=256):
        print(f"Generating {num_images} image(s) for the prompt: '{prompt}'")
        for i in range(1, num_images + 1):
            print(f"Generating image {i}/{num_images}...")
            start_time = time.time()
            
            # Generate image
            image = self.pipe(
                prompt=prompt,
                guidance_scale=0.0,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                max_sequence_length=max_sequence_length,
            ).images[0]
            
            # Save image with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"image_{timestamp}_{i}.png")
            image.save(filename)
            elapsed_time = time.time() - start_time
            print(f"Image {i} saved as {filename} (Time taken: {elapsed_time:.2f} seconds)")

        print("Image generation completed!")