import streamlit as st
from sentence_transformers import SentenceTransformer
import torch
import faiss
import docx
import PyPDF2
import numpy as np

st.title("Semantic Similarity Assessment")
st.sidebar.header("File Upload")

file1 = st.sidebar.file_uploader("Upload Document 1", type=["pdf", "docx"])
file2 = st.sidebar.file_uploader("Upload Document 2", type=["pdf", "docx"])


def extract_text(file):
    if file is not None:
        content = ""
        if file.type == "application/pdf":
            # pdf_content = file.read()
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                content += page.extract_text()
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(file)
            for para in doc.paragraphs:
                content += para.text
        return content
    return None

text1 = extract_text(file1)
text2 = extract_text(file2)

print("thats text 1",text1)
print("thats text 2",text2)


model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

def embed_text(text, model):
    if text:
        return model.encode(text, convert_to_tensor=True)
    return None

embeddings1 = embed_text(text1, model)
embeddings2 = embed_text(text2, model)

print(embeddings1, embeddings2)

@st.cache_data
def create_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

similarity_score = None

if st.sidebar.button("Calculate Similarity"):
    embeddings = np.vstack((embeddings1.cpu().numpy(), embeddings2.cpu().numpy()))
    index = create_index(embeddings)
    _, similarity_indices = index.search(embeddings, k=2)
    similarity_score = 1.0 - np.linalg.norm(embeddings[0] - embeddings[1])

st.markdown("Semantic Similarity Score")
if similarity_score != None:
    st.markdown(f"{round(similarity_score * 100, 2)}% Similarity")

