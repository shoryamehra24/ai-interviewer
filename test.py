import streamlit as st
import speech_recognition as sr

st.title("Speech-to-Text in Streamlit")

recognizer = sr.Recognizer()

if "recording" not in st.session_state:
    st.session_state.recording = False
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "transcription" not in st.session_state:
    st.session_state.transcription = None

def start_recording():
    st.session_state.recording = True
    st.session_state.transcription = None
    st.session_state.audio_data = None

def stop_recording():
    st.session_state.recording = False

# Buttons for recording
col1, col2 = st.columns(2)
with col1:
    if st.button("🎙 Start Recording"):
        start_recording()
with col2:
    if st.button("🛑 Stop Recording") and st.session_state.recording:
        stop_recording()

# Start recording when button is pressed
if st.session_state.recording:
    st.write("🔴 Recording... Speak for up to 60 seconds.")

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)  # Adjust noise settings
        try:
            st.session_state.audio_data = recognizer.listen(source, timeout=None, phrase_time_limit=60)  # No timeout, 60s limit
        except Exception as e:
            st.error(f"Recording error: {e}")

# Process transcription after stopping
if not st.session_state.recording and st.session_state.audio_data:
    try:
        st.write("🔄 Processing transcription...")
        text = recognizer.recognize_google(st.session_state.audio_data)
        st.session_state.transcription = text
        st.success(f"✅ Transcription: {text}")
    except sr.UnknownValueError:
        st.error("❌ Could not understand the audio.")
    except sr.RequestError:
        st.error("⚠️ Error connecting to the recognition service.")

# Show transcription if available
if st.session_state.transcription:
    st.write(f"📝 **Transcription:** {st.session_state.transcription}")
