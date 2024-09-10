import streamlit as st
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from langchain_groq import ChatGroq
import os

# Set page config with a colorful layout
st.set_page_config(page_title="AI Interview Assistant", layout="wide")

# Apply CSS for multi-color UI/UX with improved 3D heading
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #f9c74f, #f3722c, #f94144);
        color: #fff;
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.1); /* Translucent white background for content */
        box-shadow: 0px 15px 30px rgba(0, 0, 0, 0.2);
        border-radius: 15px;
        padding: 30px;
    }
    .title {
        font-size: 60px;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        text-shadow: 5px 5px 20px rgba(0, 0, 0, 0.7);
        margin-bottom: 20px;
        padding: 20px;
        background-color: #ff007f; /* Solid background for the heading */
        border-radius: 10px;
    }
    .subtitle {
        font-size: 22px;
        color: #f9c74f;
        text-align: center;
        margin-bottom: 40px;
        background-color: #2c2c54; /* Dark background for subtitle */
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.2);
    }
    .stButton>button {
        background-color: #ff5733;
        color: white;
        border-radius: 15px;
        font-size: 20px;
        box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.2);
    }
    .stFileUploader>label {
        font-size: 20px;
        font-weight: bold;
        color: #f9c74f;
    }
    .stMarkdown {
        background-color: rgba(255, 255, 255, 0.8); /* Light translucent background */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.2);
    }
    video {
        width: 100%;
        height: auto;
        border-radius: 15px;
        box-shadow: 0px 5px 20px rgba(0, 0, 0, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to extract audio and transcribe text
def extract_audio_and_transcribe(video_path):
    # Load video file
    video = VideoFileClip(video_path)

    # Extract audio
    audio = video.audio
    audio_path = "audio.wav"
    audio.write_audiofile(audio_path)

    # Close the video file to avoid PermissionError
    video.reader.close()
    video.audio.reader.close_proc()

    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Load the audio file
    audio_file = sr.AudioFile(audio_path)

    # Transcribe the audio
    with audio_file as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    
    return text

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key="Your api key",
    model="llama-3.1-70b-Versatile",
    temperature=0.7
)

# Function to interact with Groq API for extraction and question generation
def process_with_groq(text):
    prompt = f"""
    You are a helpful assistant that performs two tasks based on the provided text.

    1. **Extract Specific Information**:
    Extract the following details from the resume-like text:
    - Name
    - Skills
    - Projects
    - Education

    Here is the text:
    {text}

    2. **Generate Interview Questions**:
    Based on the extracted information, generate interview questions categorized by difficulty levels: easy, medium, and hard.
    """

    # Call the LLM with the prompt
    response = llm.invoke([("system", "You are a helpful assistant."), ("human", prompt)])
    
    return response.content

# Streamlit app
st.markdown('<div class="title">AI Interview Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a video to extract text and generate interview questions</div>', unsafe_allow_html=True)

# Upload video
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

# Automatically process when a video is uploaded
if uploaded_video is not None:
    # Save the uploaded video file temporarily
    video_path = f"temp_{uploaded_video.name}"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    # Display the video on the interface
    st.video(video_path)

    # Extract text from video
    with st.spinner("🎧 Extracting audio and transcribing text..."):
        transcribed_text = extract_audio_and_transcribe(video_path)
        st.write("📝 **Transcribed Text:**", transcribed_text)

    # Process with LLM
    with st.spinner("🤖 Processing with LLM..."):
        response_text = process_with_groq(transcribed_text)
        st.markdown(f"**🧠 LLM Response:**\n\n{response_text}")

    # Clean up temporary video file
    os.remove(video_path)





