import streamlit as st
import google.generativeai as genai
import time

# Configure Gemini API
genai.configure(api_key="AIzaSyD7OCGVzv-hgQw8DFXYNjQpG1Qj63KiV9w")

model = genai.GenerativeModel("gemini-2.0-flash")

# Streamlit UI
st.title("🎤 AI Mock Interview App")

# User inputs
role = st.selectbox("Select Your Role:", ["Data Analyst", "Software Engineer", "Product Manager"])
difficulty = st.selectbox("Difficulty Level:", ["Easy", "Medium", "Hard"])
num_questions = st.slider("Number of Questions:", 1, 10, 5)

if st.button("Start Interview"):
    st.session_state["questions"] = []
    st.session_state["responses"] = []
    st.session_state["answers"] = []
    st.session_state["comments"] = []  # Store comments
    st.session_state["index"] = 0
    st.session_state["interview_active"] = True

if "interview_active" in st.session_state and st.session_state["interview_active"]:
    index = st.session_state["index"]

    if index < num_questions:
        if len(st.session_state["questions"]) == index:
            prompt = f"Ask a {difficulty} level interview question for a {role} role without any additional details. Ask questions that you haven't asked previously."
            with st.spinner('🔄 Generating...'):
                model = genai.GenerativeModel("gemini-2.0-flash")
                response = model.generate_content(prompt)
            question = response.text
            st.session_state["questions"].append(question)
        else:
            question = st.session_state["questions"][index]

        st.write(f"**Question {index + 1}:** {question}")

        user_answer = st.text_area("Your Answer:", key=f"answer_{index}")
        if st.button("Submit Answer", key=f"submit_{index}"):
            st.session_state["responses"].append(user_answer)

            # Generate comment on the answer
            comment_prompt = f"Provide a short comment on this answer: {user_answer}"
            with st.spinner('🔄 Generating...'):
                model = genai.GenerativeModel("gemini-2.0-flash")
                comment_response = model.generate_content(comment_prompt)
            st.session_state["comments"].append(comment_response.text)
            st.write(f"**Comment:** {comment_response.text}")

            st.session_state["index"] += 1
            st.rerun()

        if st.button("Get Answer", key=f"get_answer_{index}"):
            answer_prompt = f"Provide an ideal answer for this question: {question}"
            with st.spinner('🔄 Generating...'):
                model = genai.GenerativeModel("gemini-2.0-flash")
                answer_response = model.generate_content(answer_prompt)
            st.session_state["answers"].append(answer_response.text)
            st.write(f"**Suggested Answer:** {answer_response.text}")

            if st.button("Next Question", key=f"next_question_{index}"):
                st.session_state["index"] += 1
                st.rerun()

        if st.button("Finish Interview"):
            st.session_state["index"] = num_questions
            st.rerun()
    else:
        st.write("✅ Interview Completed!")
        st.session_state["interview_active"] = False

        feedback_prompt = "Evaluate the following interview answers and provide concise and lenient feedback with a score out of 10, improvement pointers, and 5 resources:\n\n"
        for i, (q, a) in enumerate(zip(st.session_state['questions'], st.session_state['responses'])):
            feedback_prompt += f"Q{i+1}: {q}\nA{i+1}: {a}\n\n"

        with st.spinner('🔄 Generating Feedback...'):
            model = genai.GenerativeModel("gemini-2.0-flash")
            feedback_response = model.generate_content(feedback_prompt).text

        st.write("### 📊 AI Feedback")
        st.write(feedback_response)
