import streamlit as st
import speech_recognition as sr
import time
import numpy as np

st.title("🎤 Speech-to-Text in Streamlit")

recognizer = sr.Recognizer()

# Initialize session state variables
if "recording" not in st.session_state:
    st.session_state.recording = False
if "audio_chunks" not in st.session_state:
    st.session_state.audio_chunks = []
if "transcription" not in st.session_state:
    st.session_state.transcription = None

# Start/Stop Recording Functions
def start_recording():
    st.session_state.recording = True
    st.session_state.audio_chunks = []
    st.session_state.transcription = None

def stop_recording():
    st.session_state.recording = False

# Buttons for controlling recording
col1, col2 = st.columns(2)
with col1:
    if st.button("🎙 Start Recording"):
        start_recording()
with col2:
    if st.button("🛑 Stop Recording") and st.session_state.recording:
        stop_recording()

# Start Recording Process
if st.session_state.recording:
    st.write("🔴 Recording... Speak freely for up to 60 seconds.")

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjust for background noise
        start_time = time.time()
        max_duration = 60  # Max recording time (60 seconds)

        while time.time() - start_time < max_duration and st.session_state.recording:
            try:
                # Capture audio in 10s chunks to avoid stopping due to pauses
                audio_chunk = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                st.session_state.audio_chunks.append(audio_chunk)
            except sr.WaitTimeoutError:
                st.write("⏳ Waiting for speech...")

# Process transcription after stopping
if not st.session_state.recording and st.session_state.audio_chunks:
    st.write("🔄 Processing transcription...")
    
    # Combine all chunks into one audio file
    combined_audio = sr.AudioData(
        b"".join(chunk.get_raw_data() for chunk in st.session_state.audio_chunks),
        st.session_state.audio_chunks[0].sample_rate,
        st.session_state.audio_chunks[0].sample_width
    )

    try:
        text = recognizer.recognize_google(combined_audio)
        st.session_state.transcription = text
        st.success(f"✅ Transcription: {text}")
    except sr.UnknownValueError:
        st.error("❌ Could not understand the audio.")
    except sr.RequestError:
        st.error("⚠️ Error connecting to the recognition service.")

# Show final transcription
if st.session_state.transcription:
    st.write(f"📝 **Transcription:** {st.session_state.transcription}")
