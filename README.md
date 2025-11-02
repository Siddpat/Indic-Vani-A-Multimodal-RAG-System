# Indic-Vani: A Multimodal RAG System

Built with Krutrim's Vyakyarth Embedding Model

Indic-Vani is a multimodal (Text + Image) Retrieval-Augmented Generation (RAG) chatbot that answers questions about Indian cultural heritage. It uses a database built from Hindi text and is powered by **Krutrim's Vyakyarth model** to understand multilingual, semantic queries.

This project was built to explore and demonstrate the core technologies at Krutrim like Vyakyarth Embedding model, including RAG, multimodal search, and Gradio prototyping.

## Live Demo (GIF)

ADD VIDEO HERE

###  Architecture

This system works in two stages:

### Retrieval (The "R"): 

When you ask a query (e.g., चाहर बाग़ क्या है?):

### Text Retriever: 

Krutrim's krutrim-ai-labs/Vyakyarth model embeds the query and searches a FAISS index of Hindi text chunks to find the most relevant paragraph.

### Image Retriever: 

A multilingual CLIP model embeds the same query and searches a separate FAISS index of images to find the most relevant picture.

### Generation (The "G"):

The retrieved text chunk and the user's query are formatted into a prompt.

A generator model (**bigscience/bloomz-560m**) reads this context and generates a descriptive, natural-language answer.

The Gradio UI displays both the generated text and the retrieved image.

## How to Run

Clone the repository:

git clone https://github.com/Siddpat/Indic-Vani-A-Multimodal-RAG-System/
cd indic-vani-project



## Create a virtual environment and install dependencies:

### Create venv
python3 -m venv venv
source venv/bin/activate

### Install requirements
pip install -r requirements.txt



## (Manual Step) Add Images:
Create a folder named images in the root of the project.
Add 3-5 .jpg or .png files related to Mughal architecture (e.g., taj_mahal.jpg, charbagh_layout.jpg).

### Run the App:
python app.py



On the first run, the script will automatically scrape the Hindi Wikipedia article, create the text database, and create the image database.

On all future runs, it will skip this step and launch instantly.

### Open the App:

Open the local URL printed in your terminal (e.g., http://127.0.0.1:7860).

 
## Key Challenges & Learnings

This project highlighted several advanced RAG challenges:

## Multilingual Query Mismatch:

My biggest challenge was discovering that a query in Hinglish (e.g., charbagh) would fail, while the exact Devanagari query (चाहर बाग़) would succeed. This proves the importance of robust, cross-script tokenization and embedding—a core problem Krutrim is solving. I am still working on finding a smaller open source model that could work on Hinglish query. I tried google/flan, t5, mt5 but failed because of them being trained on English scripts.

## "Lazy" Generators:

The initial bloomz-560m generator gave very short, lazy answers (e.g., "four gardens"). By asking a more specific question ("How is it divided?"), I was able to force a more descriptive answer ("in four parts"). This demonstrates the critical role of prompt engineering and the limitations of smaller generator models. The "mistralai/Mistral-7B-Instruct-v0.2" is one of the models that was suggested by AI and also online for multilingual cases, would love to know about smaller models, if not will work on that(that does sound like a very good open source project)

Model-Specific Prompts: The T5, mT5, FLAN-T5, and BLOOMZ models all failed until fed their exact required prompt formats (e.g., fill-in-the-blank vs. instruction-based).

## 📜 License & Acknowledgements

The code for this project is licensed under the MIT License.

The embedding models are used under their respective licenses:

krutrim-ai-labs/Vyakyarth is used under the Krutrim Community License.

All other models (bloomz, clip) are used under the Apache 2.0 License.
