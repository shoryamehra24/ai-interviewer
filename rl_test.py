import streamlit as st
import google.generativeai as genai
import PyPDF2
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
        'input_ids': np.array(encoded['input_ids'], dtype=np.float32),
        'attention_mask': np.array(encoded['attention_mask'], dtype=np.float32)
    })
    
    prediction_rescaled = scaler.inverse_transform(prediction)[0]
    confidence, fluency = prediction_rescaled
    return math.ceil(confidence), math.ceil(fluency)

# === 🎯 Configure Gemini API ===
genai.configure(api_key="AIzaSyD7OCGVzv-hgQw8DFXYNjQpG1Qj63KiV9w")
model = genai.GenerativeModel("gemini-2.0-flash")

# === 🎯 Adaptive Difficulty Selection (Multi-Armed Bandit) ===
class MultiArmedBandit:
    def __init__(self, n_arms=3):
        self.n_arms = n_arms  
        self.counts = np.zeros(n_arms)  
        self.values = np.array([1.0, 1.2, 1.5])  
        self.epsilon = 0.1  

    def select_difficulty(self):
        total_counts = sum(self.counts) + 1
        ucb_values = self.values + np.sqrt(2 * np.log(total_counts) / (self.counts + 1))

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_arms)  

        return np.argmax(ucb_values)  

    def update(self, chosen_difficulty, reward):
        self.counts[chosen_difficulty] += 1
        self.values[chosen_difficulty] += (reward - self.values[chosen_difficulty]) / self.counts[chosen_difficulty]

        # **Force upgrade based on performance**  
        if reward > 0.85 and chosen_difficulty < 2:  # 85%+ score → move up a level  
            chosen_difficulty += 1  
        elif reward < 0.5 and chosen_difficulty > 0:  # <50% score → move down a level  
            chosen_difficulty -= 1  

        self.epsilon = max(0.05, 1 / (sum(self.counts) + 1))  

        return chosen_difficulty 

bandit = MultiArmedBandit()

# === 🎯 Normalize Reward Calculation ===
def calculate_reward(accuracy, confidence, fluency):
    """Normalize scores to a 0-1 range for balanced reward scaling."""
    norm_accuracy = accuracy / 100  # Scale accuracy to 0-1
    norm_confidence = confidence / 100  # Scale confidence to 0-1
    norm_fluency = fluency / 100  # Scale fluency to 0-1
    return (0.4 * norm_accuracy) + (0.2 * norm_confidence) + (0.4 * norm_fluency)

# === 🎯 Start Streamlit UI ===
if model_loaded:
    st.title("🎓 MBA Interview Practice AI")

    name = st.text_input("Candidate's Name:")
    state = st.text_input("State:")
    grad_stream = st.text_input("Graduation Stream:")
    hobbies = st.text_area("List 3 Hobbies (comma-separated):")

    resume_file = st.file_uploader("Upload Your Resume (PDF):", type=["pdf"])

    def extract_text_from_pdf(pdf_file):
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        return " ".join([page.extract_text() or "" for page in pdf_reader.pages])

    if resume_file:
        st.session_state['resume_text'] = extract_text_from_pdf(resume_file)
        st.write("📄 Resume Uploaded Successfully!")

    if st.button("Start Interview") and resume_file:
        st.session_state["index"] = 0
        st.session_state["interview_active"] = True

    if "interview_active" in st.session_state and st.session_state["interview_active"]:
        index = st.session_state["index"]

        if index < 10:
            if "current_difficulty" not in st.session_state:
                st.session_state["current_difficulty"] = 0  # Start with Easy

            difficulty_map = {0: "🟢 Easy", 1: "🟡 Medium", 2: "🔴 Hard"}
            difficulty_level = st.session_state["current_difficulty"]

            st.subheader(f"🔹 You are currently answering a **{difficulty_map[difficulty_level]}** level question")

            if "current_question" not in st.session_state:
                prompt = (
    f"Based on the following resume text and details, ask a single, relevant MBA interview question at a time without any extra context. "
    f"Limit the number of questions to a maximum of 1 per role or position mentioned in the resume. "
    f"Ensure that the questions cover different roles, positions, internships, work experience, skills, graduation stream, state, and the hobbies provided. "
    f"Additionally, ensure to include multiple questions about the latest global current affairs and latest news. "
    f"Make sure each question is clear, concise, and focuses on one specific topic without combining multiple questions. "
    f"The order of questions should be random.\n\n"

    f"Adjust the difficulty of the question based on the candidate’s previous performance:\n"
    f"- If the candidate’s responses have been highly accurate, confident, and fluent, generate a complex question requiring deep analytical thinking, strategic decision-making, or case-based reasoning.\n"
    f"- If the candidate’s responses have been moderately strong, ask a balanced question requiring conceptual understanding and practical application.\n"
    f"- If the candidate’s responses have been weaker, ask simpler, more direct questions to help them build confidence before increasing complexity.\n\n"

    f"Resume:\n{st.session_state['resume_text']}\n\n"
    f"Details:\nName: {name}\nState: {state}\nGraduation Stream: {grad_stream}\nHobbies: {hobbies}\n\n"
    
    f"Current Difficulty Level: {difficulty_map[difficulty_level]}"
)


                with st.spinner('🔄 Generating Question...'):
                    response = model.generate_content(prompt)
                
                st.session_state["current_question"] = response.text.strip()

            st.write(f"**Question {index + 1}:** {st.session_state['current_question']}")

            user_answer = st.text_area("Your Answer:", key=f"answer_{index}")

            if st.button("Submit Answer", key=f"submit_{index}"):
                if user_answer.strip():
                    confidence, fluency = predict_confidence_fluency(user_answer)

                    accuracy_prompt = (
                        f"Evaluate the accuracy of this answer and give a score out of 100. Respond only with a single integer number, nothing else.\n\n"
                        f"Question: {st.session_state['current_question']}\nAnswer: {user_answer}"
                    )

                    with st.spinner('🔄 Calculating Accuracy...'):
                        accuracy_response = model.generate_content(accuracy_prompt)

                    try:
                        accuracy_score = int(accuracy_response.text.strip().split()[0])
                    except ValueError:
                        accuracy_score = 50  

                    st.session_state["confidence"] = confidence
                    st.session_state["fluency"] = fluency
                    st.session_state["accuracy"] = accuracy_score
                    st.session_state["submitted"] = True

            if st.session_state.get("submitted", False):
                st.write("### 📊 AI Evaluation")
                st.write(f"🔵 **Confidence:** {st.session_state['confidence']}/100")
                st.write(f"🟢 **Fluency:** {st.session_state['fluency']}/100")
                st.write(f"🟠 **Accuracy:** {st.session_state['accuracy']}/100")

                if st.button("Next Question", key=f"next_question_{index}"):
                    reward = calculate_reward(st.session_state["accuracy"], st.session_state["confidence"], st.session_state["fluency"])
                    st.session_state["current_difficulty"] = bandit.update(difficulty_level, reward)

                    st.session_state["index"] += 1
                    st.session_state.pop("current_question", None)  
                    st.session_state.pop("submitted", None)  
                    st.rerun()

            if st.button("Finish Interview"):
                st.session_state["interview_active"] = False
                st.rerun()
        else:
            st.write("✅ Interview Completed!")
            st.session_state["interview_active"] = False
