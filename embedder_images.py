from transformers import AutoProcessor, AutoModel
from PIL import Image
import glob
import numpy as np
import os
import torch

def create_image_embeddings():
    """
    Finds all images in the 'images/' folder,
    embeds them using the ORIGINAL CLIP model,
    and saves the embeddings and file paths.
    """
    
    # 1. Load the ORIGINAL OpenAI CLIP Model and Processor
    # This is the correct model for encoding images.
    model_name = "openai/clip-vit-base-patch32"
    
    print(f"Loading CLIP model and processor: {model_name}...")
    
    # Use AutoProcessor, which will correctly find the processor
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    print("CLIP model and processor loaded.")

    image_folder = 'images/'
    
    # 2. Find all images
    image_paths = list(glob.glob(f'{image_folder}*.jpg'))
    image_paths.extend(list(glob.glob(f'{image_folder}*.png')))
    image_paths.extend(list(glob.glob(f'{image_folder}*.jpeg')))

    if not image_paths:
        print(f"Error: No images found in '{image_folder}' folder.")
        print("Please download some .jpg files and put them there.")
        return

    print(f"Found {len(image_paths)} images. Opening them...")
    
    # 3. Open images
    pil_images = [Image.open(path) for path in image_paths]
    
    # 4. Create Embeddings
    print("Creating embeddings for all images...")
    
    # Process the images
    # We remove padding=True as it's not needed here and caused a warning
    inputs = processor(images=pil_images, return_tensors="pt")

    # Get the image embeddings
    with torch.no_grad():
        image_embeddings = model.get_image_features(**inputs)
        
    # Normalize the embeddings
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    
    image_embeddings_np = image_embeddings.cpu().numpy()

    print("Image embeddings created.")

    # 5. Save the results
    np.save('image_embeddings.npy', image_embeddings_np)
    
    with open('image_files.txt', 'w', encoding='utf-8') as f:
        for path in image_paths:
            f.write(path + '\n')

    print(f"Successfully saved embeddings to 'image_embeddings.npy'")
    print(f"Successfully saved file paths to 'image_files.txt'")

if __name__ == "__main__":
    if os.path.exists('image_embeddings.npy'):
        os.remove('image_embeddings.npy')
    if os.path.exists('image_files.txt'):
        os.remove('image_files.txt')
        
    create_image_embeddings()