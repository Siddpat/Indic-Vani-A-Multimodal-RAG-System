import nltk
from sentence_transformers import SentenceTransformer
import numpy as np
import warnings
import os

# --- 1. SETUP ---
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data")

def process_text_file(file_name):
    """
    Reads, chunks (by paragraph!), and embeds the text from a given file.
    """
    print(f"--- Starting Step 2 (v4): Paragraph Chunking ---")

    # --- 2. READ THE DATA ---
    print(f"Reading text from {file_name}...")
    with open(file_name, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # --- *** NEW, SIMPLER CHUNKING LOGIC *** ---
    print("Chunking text into paragraphs...")
    
    # Split the text by newline characters
    chunks = text.split('\n')
    
    # Clean up the list: remove empty lines and very short scraps
    chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 50]
    
    print(f"Created {len(chunks)} paragraph chunks.")
    if chunks:
        print("Example chunk:", chunks[0])
    else:
        print("No chunks were created. Check the text file.")
        return

    # --- 4. EMBEDDING ---
    model_name = "krutrim-ai-labs/Vyakyarth"
    print(f"\nLoading embedding model: '{model_name}'...")
    model = SentenceTransformer(model_name)
    print("Model loaded.")

    print("Creating embeddings for all chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    print("Embeddings created successfully!")

    # --- 5. SAVING ---
    np.save("embeddings.npy", embeddings)
    with open("chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n")
            
    print("\nSuccessfully saved new chunks to 'chunks.txt' and embeddings to 'embeddings.npy'")
    print("--- Step 2 (v4) Complete ---")

if __name__ == "__main__":
    
    if os.path.exists("embeddings.npy"):
        os.remove("embeddings.npy")
    if os.path.exists("chunks.txt"):
        os.remove("chunks.txt")
    print("Removed old chunk and embedding files.")
    
    # Make sure we're using the HINDI file
    process_text_file("mughal_architecture_hindi.txt")