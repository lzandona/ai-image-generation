import torch
import intel_extension_for_pytorch as ipex
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import time
from datetime import datetime
from pathlib import Path

class StableDiffusionGenerator:

    def __init__(self, model_name="stabilityai/stable-diffusion-2-1-base", output_dir="output"):
        """
        Initialize the Stable Diffusion generator.

        Args:
            model_name (str): The model to load from the Hugging Face hub.
            output_dir (str): Directory to save generated images.
        """

        self.device = torch.device("xpu") if torch.xpu.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")

        print("Loading Stable Diffusion pipeline...")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_name)

        # Set the sampler to DPM++ 2M
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        print("Using DPM++ 2M Sampler")

        self.pipe.to(self.device)

        # Optimize the model for XPU
        if self.device.type == "xpu":
            self.pipe.unet = ipex.optimize(self.pipe.unet, dtype=torch.float32)
            self.pipe.vae = ipex.optimize(self.pipe.vae, dtype=torch.float32)
            self.pipe.text_encoder = ipex.optimize(self.pipe.text_encoder, dtype=torch.float32)

        # Create output folder if it doesn't exist
        self.output_folder = Path(output_dir)
        self.output_folder.mkdir(exist_ok=True)
        print(f"Saving images to folder: {self.output_folder.resolve()}")

    def generate_images(
            self,
            prompt,
            negative_prompt="",
            num_inference_steps=28,
            guidance_scale=4.5,
            height=512,
            width=512,
            num_images=1,
    ):
        """
        Generate images based on the provided prompt.

        Args:
            prompt (str): The main text prompt for generation.
            negative_prompt (str): The negative prompt to discourage undesired attributes.
            num_inference_steps (int): Number of inference steps for the generation process.
            guidance_scale (float): Guidance scale for prompt adherence.
            height (int): Image height.
            width (int): Image width.
            num_images (int): Number of images to generate.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i in range(num_images):
            start_time = time.time()
            print(f"Generating image {i + 1} of {num_images}...")

            # Generate the image
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]

            end_time = time.time()

            # Save the image with timestamp and counter
            output_path = self.output_folder / f"output_xpu_{timestamp}_{i + 1}.png"
            image.save(output_path)
            print(f"Image {i + 1} saved at {output_path}")
            print(f"Image generation completed in {end_time - start_time:.2f} seconds.")               
        


if __name__ == "__main__":
    # Initialize the generator
    generator = StableDiffusionGenerator(model_name="stabilityai/stable-diffusion-2-1-base")

    # Set parameters for text-to-image generation
    prompt = "A dog holding a hello world sign"
    negative_prompt = "blurry, cartoonish, low quality, distorted details, watermark"
    num_images = 1

    # Generate images
    generator.generate_images(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=28,
        guidance_scale=4.5,
        height=512,
        width=512,
        num_images=num_images,
    )

