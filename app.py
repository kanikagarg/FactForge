import dotenv
import streamlit as st
import os
import random
from utils import generate_options

dotenv.load_dotenv()
if os.environ["HF_TOKEN"]:
    print("HF_TOKEN FOUND")

from datasets import load_dataset
queans = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
knowledge = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
# k = 10 number of questions to ask
k = 10
indices = random.sample(range(len(queans['test'])-1), k=k)
st.title("📝 Knowledge Test ")

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
        print("Generating options...")
        options_str = generate_options(question_text, correct_answer)
        print("Options generated:", options_str)
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
            st.error(f"Q{i+1}: Incorrect | Correct answer: {q_data['correct_answer']}", icon="🚨")

    st.divider()
    st.metric("Final Score", f"{score} / {k}")