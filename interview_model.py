import streamlit as st
import google.generativeai as genai
import PyPDF2
import time
import numpy as np
import joblib
import math
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, TFBertModel

st.set_page_config(layout="centered")

# === 🎯 Load AI Scoring Model (Confidence & Fluency) ===
@st.cache_resource
def load_ai_models():
    try:
        scoring_model = load_model("saved_model/bert_interview_scorer", custom_objects={"TFBertModel": TFBertModel})
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        scaler = joblib.load("saved_model/scaler.pkl")
        return scoring_model, tokenizer, scaler
    except Exception as e:
        st.error(f"⚠️ Error loading AI model: {e}")
        return None, None, None

scoring_model, tokenizer, scaler = load_ai_models()
model_loaded = scoring_model is not None
# === 🎯 Function to Predict Confidence & Fluency ===
def predict_confidence_fluency(response):
    encoded = tokenizer(
        [response],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="tf"
    )
    
    prediction = scoring_model.predict({
        'input_ids': np.array(encoded['input_ids']),
        'attention_mask': np.array(encoded['attention_mask'])
    })
    
    prediction_rescaled = scaler.inverse_transform(prediction)[0]
    confidence, fluency = prediction_rescaled
    return math.ceil(confidence), math.ceil(fluency)

# === 🎯 Configure Gemini API ===
genai.configure(api_key="AIzaSyD7OCGVzv-hgQw8DFXYNjQpG1Qj63KiV9w")
model = genai.GenerativeModel("gemini-2.0-flash")

# === ✅ Start Streamlit UI Only If Model is Loaded ===
if model_loaded:
    
    st.title("🎓 MBA Interview Practice AI")

    # === 🎯 User Inputs ===
    name = st.text_input("Candidate's Name:")
    state = st.text_input("State:")
    grad_stream = st.text_input("Graduation Stream:")
    hobbies = st.text_area("List 3 Hobbies (comma-separated):")

    # === 🎯 Resume Upload (PDF) ===
    resume_file = st.file_uploader("Upload Your Resume (PDF):", type=["pdf"])

    def extract_text_from_pdf(pdf_file):
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        resume_text = ""
        for page in pdf_reader.pages:
            resume_text += page.extract_text()
        return resume_text

    if resume_file:
        resume_text = extract_text_from_pdf(resume_file)
        st.session_state['resume_text'] = resume_text
        st.write("📄 Resume Uploaded Successfully!")

    # === 🎯 Start Interview Button ===
    if st.button("Start Interview") and resume_file:
        st.session_state["questions"] = []
        st.session_state["responses"] = []
        st.session_state["answers"] = []
        st.session_state["comments"] = []
        st.session_state["index"] = 0
        st.session_state["interview_active"] = True
        st.session_state["self_intro"] = False  # Flag for self-introduction

    # === 🎯 Interview Flow ===
    if "interview_active" in st.session_state and st.session_state["interview_active"]:
        index = st.session_state["index"]

        # === Self Introduction ===
        if not st.session_state["self_intro"]:
            st.write(f"Hello {name}, welcome to your MBA mock interview! I see you're from {state} and pursued {grad_stream}. Let's begin!")
            st.write("Before we begin, could you please introduce yourself?")
            
            user_intro = st.text_area("Your Introduction:", key="self_intro_input")
            if st.button("Submit Introduction"):
                st.session_state["self_intro"] = True
                st.session_state["responses"].append(user_intro)
                st.write("Thank you! Let's proceed to the interview questions.")
                st.session_state["index"] = 0

        # === Main Interview Questions ===
        if st.session_state["self_intro"]:
            if index < 10:
                if len(st.session_state["questions"]) == index:
                    prompt = (
                        f"Based on the resume and provided details, generate a single, relevant MBA interview question.\n\n"
                        f"Resume:\n{st.session_state['resume_text']}\n\n"
                        f"Details:\nName: {name}\nState: {state}\nGraduation Stream: {grad_stream}\nHobbies: {hobbies}"
                    )

                    with st.spinner('🔄 Generating Question...'):
                        response = model.generate_content(prompt)
                    question = response.text.strip()
                    st.session_state["questions"].append(question)
                else:
                    question = st.session_state["questions"][index]

                st.write(f"**Question {index + 1}:** {question}")

                user_answer = st.text_area("Your Answer:", key=f"answer_{index}")

                if st.button("Submit Answer", key=f"submit_{index}"):
                    st.session_state["responses"].append(user_answer)

                    # === 🎯 AI Model Scores (Confidence & Fluency) ===
                    confidence, fluency = predict_confidence_fluency(user_answer)

                    # === 🎯 Gemini Accuracy Evaluation ===
                    accuracy_prompt = f"Evaluate the accuracy of this answer based on the given question and give a score out of 100:\n\n"
                    accuracy_prompt += f"Question: {question}\nAnswer: {user_answer}\n\n"
                    accuracy_prompt += "Only provide a numerical accuracy score out of 100, nothing else."

                    with st.spinner('🔄 Calculating Accuracy...'):
                        accuracy_response = model.generate_content(accuracy_prompt)
                    accuracy_score = accuracy_response.text.strip()

                    # Display Scores
                    st.write("### 📊 AI Evaluation")
                    st.write(f"🔵 **Confidence:** {confidence}/10")
                    st.write(f"🟢 **Fluency:** {fluency}/10")
                    st.write(f"🟠 **Accuracy:** {accuracy_score}/100")

                    # === 🎯 Generate AI Comment ===
                    comment_prompt = f"Provide a short and concise comment on this answer: {user_answer}"
                    with st.spinner('🔄 Generating Comment...'):
                        comment_response = model.generate_content(comment_prompt)
                    st.session_state["comments"].append(comment_response.text.strip())
                    st.write(f"**Comment:** {comment_response.text.strip()}")

                # === Suggested Answer Button ===
                if st.button("Get Suggested Answer", key=f"get_answer_{index}"):
                    answer_prompt = f"Provide an ideal but concise answer for this question: {question}"
                    with st.spinner('🔄 Generating Suggested Answer...'):
                        answer_response = model.generate_content(answer_prompt)
                    st.session_state["answers"].append(answer_response.text.strip())
                    st.write(f"**Suggested Answer:** {answer_response.text.strip()}")

                # === Next Question Button ===
                if st.button("Next Question", key=f"next_question_{index}"):
                    st.session_state["index"] += 1
                    st.rerun()

                # === Finish Interview Button ===
                if st.button("Finish Interview"):
                    st.session_state["interview_active"] = False
                    st.rerun()
            else:
                st.write("✅ Interview Completed!")
                st.session_state["interview_active"] = False

            # Feedback generation based on answered questions
            feedback_prompt = "Evaluate the following interview answers and provide concise feedback with a score out of 10, improvement pointers, and 5 resources:\n\n"
            for i, (q, a) in enumerate(zip(st.session_state['questions'], st.session_state['responses'])):
                feedback_prompt += f"Q{i+1}: {q}\nA{i+1}: {a}\n\n"

            with st.spinner('🔄 Generating Feedback...'):
                feedback_response = model.generate_content(feedback_prompt).text

            st.write("### 📊 AI Feedback")
            st.write(feedback_response)

    else:
        if not resume_file:
            st.warning("⚠️ Please upload your resume to start the interview.")

