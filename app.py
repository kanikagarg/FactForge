import dotenv
import streamlit as st
import os
import random
from utils import generate_options
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss    
from datasets import load_dataset
dotenv.load_dotenv()

queans = None
knowledge = None
k = 10
indices = None
st.cache_data(show_spinner="Loading dataset...")
def load_datasets():
    global queans, knowledge, indices
    queans = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
    knowledge = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
    indices = random.sample(range(len(queans['test'])-1), k=k)

load_datasets()

# Process knowledge for RAG
@st.cache_resource(show_spinner="Loading dataset...")
def load_knowledge_base():
    # Load embedding model
    global knowledge
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Get passages - assuming 'passage' field in the dataset
    passages = knowledge['passages']
    
    # Create embeddings
    embeddings = model.encode(passages, convert_to_tensor=False)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    faiss.normalize_L2(embeddings)  # Normalize for cosine
    index.add(embeddings)
    
    return model, index, passages

model, index, passages = load_knowledge_base()

def retrieve_relevant_passages(query, top_k=2):
    query_embedding = model.encode([query], convert_to_tensor=False)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    results = [passages[i] for i in indices[0]]
    return results

st.title("📝 Knowledge Test & Knowledge Base")

# Tab management with session state
tab_names = ["📝 Quiz", "📚 Knowledge Base", "🔍 Search Knowledge"]
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0

# Sidebar for tab selection
selected_tab = st.sidebar.selectbox("Select Section", tab_names, index=st.session_state.active_tab)

# Update active tab
st.session_state.active_tab = tab_names.index(selected_tab)

# Display content based on selected tab
if selected_tab == "📝 Quiz":
    st.header("Knowledge Test")
    st.write("Answer the following questions:")

    if "responses" not in st.session_state:
        st.session_state.responses = {}

    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    # Initialize questions and options only once
    if "questions_data" not in st.session_state:
        indices = random.sample(range(len(queans['test'])-1), k=k)
        st.session_state.questions_data = []
        for i, q in enumerate(indices):
            question_text = queans['test'][q]['question']
            correct_answer = queans['test'][q]['answer']
            # print("Generating options...")
            options_str = generate_options(question_text, correct_answer)
            # print("Options generated:", options_str)
            options = [option.strip() for option in options_str.split("/")] + [correct_answer]
            random.shuffle(options)
            st.session_state.questions_data.append({
                'question': question_text,
                'correct_answer': correct_answer,
                'options': options
            })

    # -----------------------
    # Display questions
    # -----------------------
    for i, q_data in enumerate(st.session_state.questions_data):
        st.subheader(f"Q{i+1}. {q_data['question']}")
        # Question is displayed
        # Now create 3 choices relevant to the questions, but not the correct answer
        selected = st.radio(
            "Choose one:",
            q_data['options'],
            key=f"question_{i}",
            index=None if f"question_{i}" not in st.session_state.responses else
            q_data['options'].index(st.session_state.responses[f"question_{i}"]),
        )
        if selected:
            st.session_state.responses[f"question_{i}"] = selected

    # -----------------------
    # Submit button
    # -----------------------
    all_answered = all(f"question_{i}" in st.session_state.responses for i in range(len(st.session_state.questions_data)))
    if st.button("Submit Test", disabled=not all_answered):
        st.session_state.submitted = True

    # -----------------------
    # Results
    # -----------------------
    if st.session_state.submitted:
        score = 0

        st.divider()
        st.subheader("📊 Results")

        for i, q_data in enumerate(st.session_state.questions_data):
            user_answer = st.session_state.responses.get(f"question_{i}")

            if user_answer == q_data['correct_answer']:
                score += 1
                st.success(f"Q{i+1}: Correct")
            else:
                # emb = get_correct_answer_embedding(q_data['question'])
                st.error(f"Q{i+1}: Incorrect | Correct answer: {q_data['correct_answer']}", icon="🚨")
                # st.write(f"Embedding of correct answer: {emb}")

        st.divider()
        st.metric("Final Score", f"{score} / {k}")

elif selected_tab == "📚 Knowledge Base":
    st.header("📚 Knowledge Base")
    st.write("Browse the knowledge passages from the dataset.")
    
    passages_per_page = 10
    total_passages = len(passages)
    total_pages = (total_passages + passages_per_page - 1) // passages_per_page
    
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start_idx = (page - 1) * passages_per_page
    end_idx = min(start_idx + passages_per_page, total_passages)
    
    st.write(f"Showing passages {start_idx + 1} to {end_idx} of {total_passages}")
    
    for i in range(start_idx, end_idx):
        st.subheader(f"Passage {i + 1}")
        st.write(passages[i])
        st.divider()

elif selected_tab == "🔍 Search Knowledge":
    st.header("🔍 Search Knowledge")
    st.write("Enter a query to find relevant information from the knowledge base.")
    
    query = st.text_input("Enter your query:")
    if st.button("Search"):
        if query:
            results = retrieve_relevant_passages(query, top_k=5)
            st.subheader("Relevant Passages:")
            for idx, passage in enumerate(results):
                st.write(f"**Passage {idx + 1}:**")
                st.write(passage)
                st.divider()
        else:
            st.warning("Please enter a query.")