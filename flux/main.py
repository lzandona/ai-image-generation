from image_generator import ImageGenerator

def main():
    # Initialize the image generator
    generator = ImageGenerator()

    # Get user input
    prompt = input("Enter the prompt for the image generation: ").strip()
    while not prompt:
        print("Prompt cannot be empty.")
        prompt = input("Enter the prompt for the image generation: ").strip()


    num_images = int(input("Enter the number of images to generate: "))
    if num_images <= 0:
        print("Invalid input for the number of images. Defaulting to 1 image.")
        num_images = 1

    # Generate images
    generator.generate_images(prompt=prompt, num_images=num_images)


if __name__ == "__main__":
    main()