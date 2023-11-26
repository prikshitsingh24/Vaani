from PIL import Image
import os

def create_gif(directory_path, output_path, gif_name='output.gif', fps=2,):
    """
    Create a smooth GIF from a set of images in a directory.

    Parameters:
    - directory_path: Path to the directory containing images.
    - output_path: Path to save the output GIF.
    - gif_name: Name of the output GIF file (default is 'output.gif').
    - fps: Frames per second for the GIF (default is 24).
    - loop: Number of loops for the GIF (0 for infinite loop, default is 0).
    """
    images = []
    
    # Get a list of image files in the directory
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    image_files.sort()  # Sort the files to maintain order
    
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        img = Image.open(image_path)
        img = img.resize((img.width * 2, img.height * 2), Image.ANTIALIAS)  # Resize for smoother animation
        images.append(img)

    # Save the GIF
    output_gif_path = os.path.join(output_path, gif_name)
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),  # Calculate duration in milliseconds
        
    )
    print(f"GIF created successfully at {output_gif_path}")

