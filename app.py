import gradio as gr
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import torch
import os

# --- 1. SETUP & HELPER FUNCTIONS ---
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data")
torch.set_grad_enabled(False)

def load_text_data():
    embeddings = np.load("embeddings.npy")
    with open("chunks.txt", "r", encoding="utf-8") as f:
        chunks = f.read().splitlines()
    return chunks, embeddings.astype('float32') 

def build_faiss_index(embeddings):
    d = embeddings.shape[1] 
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index

def load_text_retriever_model():
    model_name = "krutrim-ai-labs/Vyakyarth"
    model = SentenceTransformer(model_name)
    return model

def search_text_index(query, model, index, chunks, k=1):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding.astype('float32'), k)
    return [chunks[i] for i in indices[0]]

def load_image_data():
    image_embeddings = np.load("image_embeddings.npy")
    with open("image_files.txt", "r", encoding="utf-8") as f:
        image_files = f.read().splitlines()
    return image_files, image_embeddings.astype('float32')

def load_image_query_model():
    model_name = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    model = SentenceTransformer(model_name)
    return model

def search_image_index(query, model, index, image_files, k=1):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding.astype('float32'), k)
    return [image_files[i] for i in indices[0]]

# --- *** REVERTED TO BLOOMZ *** ---
def load_generation_model():
    """
    Loads the BLOOMZ-560m model.
    """
    print("Loading text generation model (bigscience/bloomz-560m)...")
    model_name = "bigscience/bloomz-560m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("Generation model and tokenizer loaded.")
    return model, tokenizer

# --- *** REVERTED TO BLOOMZ PROMPT *** ---
def build_prompt(query, context_chunks):
    """
    Builds the Hindi-instruction prompt for BLOOMZ.
    """
    context_str = " ".join(context_chunks)
    prompt = f"""
कृपया संदर्भ के आधार पर प्रश्न में दिए गए शब्द का वर्णन करें।

संदर्भ (Context):
{context_str}

प्रश्न (Question):
{query}

उत्तर (Answer):
"""
    return prompt

# --- 2. LOAD ALL MODELS ONCE ---
print("Loading all models... This will take a moment.")

print("Loading text database...")
text_chunks, text_embeddings = load_text_data()
text_index = build_faiss_index(text_embeddings)
text_retriever = load_text_retriever_model()

print("Loading image database...")
image_files, image_embeddings = load_image_data()
image_index = build_faiss_index(image_embeddings)
image_retriever = load_image_query_model()

print("Loading generator model (Bloomz)...")
gen_model, gen_tokenizer = load_generation_model()

print("--- System Ready ---")

# --- 3. THE "MASTER" RAG FUNCTION ---
def run_rag_pipeline(query):
    # 1. Retrieve Text
    context_chunks = search_text_index(query, text_retriever, text_index, text_chunks, k=1)
    
    # 2. Retrieve Image
    image_path = search_image_index(query, image_retriever, image_index, image_files, k=1)
    
    # 3. Build Prompt
    prompt = build_prompt(query, context_chunks)
    
    # --- *** REVERTED TO BLOOMZ GENERATION *** ---
    # 4. Generate Text
    inputs = gen_tokenizer(prompt, return_tensors="pt")
    output_ids = gen_model.generate(
        inputs.input_ids, 
        attention_mask=inputs.attention_mask,
        max_new_tokens=100, 
        num_beams=4,
        early_stopping=True,
        pad_token_id=gen_tokenizer.eos_token_id
    )
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    final_answer = gen_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    # 5. Return the two outputs
    # Gradio is smart and knows to render the image path
    return final_answer, image_path[0]


# --- 4. BUILD THE GRADIO UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Indic-Vani: Multimodal RAG System")
    gr.Markdown("Ask a question about Mughal architecture in Hindi (e.g., *चाहर बाग़ क्या है?*)")

    with gr.Row():
        query_box = gr.Textbox(
            label="Your Question", 
            placeholder="चाहर बाग़ को कैसे विभाजित किया जाता है?",
            lines=2
        )
    
    submit_btn = gr.Button("Submit")

    with gr.Row():
        answer_box = gr.Textbox(label="Generated Answer", lines=6)
        image_box = gr.Image(label="Retrieved Image")

    submit_btn.click(
        fn=run_rag_pipeline,
        inputs=query_box,
        outputs=[answer_box, image_box]
    )
    
    gr.Examples(
        examples=[
            ["चाहर बाग़ क्या है?"],
            ["चाहर बाग़ को कैसे विभाजित किया जाता है?"],
            ["ताज महल किसने बनवाया?"]
        ],
        inputs=query_box
    )

# --- 5. LAUNCH THE APP ---
print("Launching Gradio app... Open the URL in your browser.")
demo.launch()