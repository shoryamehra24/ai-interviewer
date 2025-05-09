import streamlit as st
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyD7OCGVzv-hgQw8DFXYNjQpG1Qj63KiV9w")

st.set_page_config(layout="centered")  # Center-align UI
st.title("🎤 AI Mock Interview App")

# User inputs
role = st.selectbox("Select Your Role:", ["Data Analyst", "Software Engineer", "Product Manager"])
difficulty = st.selectbox("Difficulty Level:", ["Easy", "Medium", "Hard"])
num_questions = st.slider("Number of Questions:", 1, 10, 5)

# Initialize session state
if st.button("Start Interview"):
    st.session_state["questions"] = []
    st.session_state["responses"] = []
    st.session_state["index"] = 0
    st.session_state["follow_ups"] = []
    st.session_state["answers"] = []
    st.session_state["interview_active"] = True

if "interview_active" in st.session_state and st.session_state["interview_active"]:
    index = st.session_state["index"]

    if index < num_questions:
        if len(st.session_state["questions"]) == index:
            # Generate new question
            prompt = f"Ask a {difficulty} level interview question for a {role} role without any additional details. Ask questions that you haven't asked previously."
            model = genai.GenerativeModel("gemini-2.0-flash")
            
            with st.spinner('🔄 Generating...'):
                response = model.generate_content(prompt)
            
            question = response.text if hasattr(response, 'text') else (
                response.candidates[0].text if response.candidates else "Error: No response generated.")
            
            st.session_state["questions"].append(question)
            st.session_state["follow_ups"].append("")
            st.session_state["answers"].append("")
        else:
            question = st.session_state["questions"][index]

        st.write(f"**Question {index + 1}:** {question}")
        user_answer = st.text_area("Your Answer:", key=f"answer_{index}")

        if st.button("Submit Answer", key=f"submit_{index}"):
            if len(st.session_state["responses"]) <= index:
                st.session_state["responses"].append(user_answer)
            else:
                st.session_state["responses"][index] = user_answer

            follow_up_prompt = f"Provide a follow-up question for this response:\nQ: {question}\nA: {user_answer}"
            
            with st.spinner('🔄 Generating...'):
                model = genai.GenerativeModel("gemini-2.0-flash")
                follow_up_response = model.generate_content(follow_up_prompt)
            
            st.session_state["follow_ups"][index] = follow_up_response.text
            st.rerun()

        if st.session_state["follow_ups"][index]:
            st.write(f"**Follow-up Question:** {st.session_state['follow_ups'][index]}")

        if st.button("Get Answer", key=f"get_answer_{index}"):
            answer_prompt = f"Provide an ideal answer for this question:\n{question}"
            
            with st.spinner('🔄 Generating...'):
                model = genai.GenerativeModel("gemini-2.0-flash")
                answer_response = model.generate_content(answer_prompt)

            
            st.session_state["answers"][index] = answer_response.text
            st.rerun()

        if st.session_state["answers"][index]:
            st.write(f"**Suggested Answer:** {st.session_state['answers'][index]}")
            
            # Show 'Next Question' button after displaying the answer
            if st.button("Next Question", key=f"next_question_{index}"):
                st.session_state["index"] += 1
                st.rerun()

        if st.button("Finish Interview"):
            st.session_state["index"] = num_questions
            st.rerun()
    
    else:
        st.write("✅ Interview Completed!")
        st.session_state["interview_active"] = False

        feedback_prompt = "Evaluate the following interview answers and provide feedback with a score out of 10, improvement pointers, and resources:\n\n"
        for i, (q, a) in enumerate(zip(st.session_state['questions'], st.session_state['responses'])):
            feedback_prompt += f"Q{i+1}: {q}\nA{i+1}: {a}\n\n"
        
        with st.spinner('🔄 Generating Feedback...'):
            feedback_response = genai.GenerativeModel("gemini-2.0-flash").generate_content(feedback_prompt).text
        
        st.write("### 📊 AI Feedback")
        st.write(feedback_response)
