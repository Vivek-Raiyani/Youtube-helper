import streamlit as st
from utils import *
import json


# ==========================
# Streamlit Page Configuration
# ==========================
st.set_page_config(
    page_title="My Streamlit App",
    page_icon="🏠",
    layout="wide"
)

# ==========================
# Sidebar Inputs
# ==========================
st.title("Welcome to My App")

# API Key input
st.session_state['api_key'] = st.sidebar.text_input(
    label="Enter your Samba Nova API key:",
    type="password", 
)

# Model selection
st.session_state['model'] = st.sidebar.radio(
    "Choose Model",
    [
        "Meta-Llama-3.1-8B-Instruct",
        "Meta-Llama-3.1-70B-Instruct",
        "Meta-Llama-3.1-405B-Instruct",
        "Meta-Llama-3.2-1B-Instruct",
        "Meta-Llama-3.2-3B-Instruct"
    ]
)

# YouTube video link input
new_video_url = st.sidebar.text_input(label="YouTube video link:")

# ==========================
# Clear Cache for New URL
# ==========================
if 'link1' not in st.session_state or st.session_state['link1'] != new_video_url:
    # Clear previous session data
    st.session_state.pop('transcripts', None)
    st.session_state.pop('full_text', None)
    st.session_state.pop('vectorstore', None)

    # Update session with the new video URL
    st.session_state['link1'] = new_video_url

# ==========================
# Warn if no video link is provided
# ==========================
if not st.session_state['link1']:
    st.warning("Please provide a YouTube video URL.")

# ==========================
# Main Functionality
# ==========================
else:
    try:
        # Fetch transcript and full text if not already loaded
        if 'transcripts' not in st.session_state and 'full_text' not in st.session_state:
            full_text_video1, transcript_video1 = get_transcripts(st.session_state['link1'])
            
            if not full_text_video1:
                st.error("Failed to fetch transcript. Please check the video URL.")
                st.stop()
            
            st.session_state['full_text'] = full_text_video1
            st.session_state['transcripts'] = json.dumps(transcript_video1)
        
        col1 ,col2 =st.columns(2,gap='small')
        with col1:
            st.video(st.session_state['link1'])
        with col2:
            # User selects analysis type
            st.session_state.analysis_type = st.radio(
            "Choose Analysis Type",
            [
                "Summarize",
                "Generate Quiz",
                "Q&A from Video",
                "Find Short-Form Timestamps",
                "Generate Blog Post"
            ]
            )
        
        

        # ==========================
        # Summarization
        # ==========================
        if st.session_state.analysis_type == "Summarize":
            st.subheader("Video Summary")
            if 'summary' not in st.session_state:
                st.session_state['summary'] = generate_summary(
                    st.session_state['full_text'],
                    st.session_state['api_key'],
                    model=st.session_state['model']
                )
            st.write(st.session_state['summary'])

        # ==========================
        # Find Short-Form Timestamps
        # ==========================
        elif st.session_state.analysis_type == "Find Short-Form Timestamps":
            st.subheader("Recommended Short-Form Segments")
            st.info("Tip: Use Meta-Llama-3.1-70B-Instruct for better results.")
            transcript = json.loads(st.session_state['transcripts'])
            chunks = find_short_form_timestamps(
                transcript=transcript,
                api_key=st.session_state['api_key'],
                model=st.session_state['model']
            )
            if chunks:
                for chunk in chunks:
                    st.write(chunk)
            else:
                st.warning("No short-form timestamps were identified.")

        # ==========================
        # Generate Quiz
        # ==========================
        elif st.session_state.analysis_type == "Generate Quiz":
            st.subheader("Video Quiz")
            quiz = generate_quiz(
                st.session_state['full_text'],
                st.session_state['api_key'],
                num_questions=10,
                model=st.session_state['model']
            )
            if quiz:
                st.write(quiz)
            else:
                st.error("Failed to generate quiz. Please verify your API key and model settings.")

        # ==========================
        # Q&A from Video
        # ==========================
        elif st.session_state.analysis_type == "Q&A from Video":
            st.subheader("Video Question & Answer")
            st.info("Tip: Use Meta-Llama-3.2-1B-Instruct for better results.")
            
            # Split text and setup RAG
            text_chunks = split_text(st.session_state['full_text'])
            qa_chain, vectorstore = setup_rag(
                text_chunks=text_chunks,
                api_key=st.session_state['api_key'],
                model=st.session_state['model']
            )
            
            if qa_chain:
                question = st.text_input("Ask a question about the video.")
                if st.button("Get Answer"):
                    if question:
                        answer = qa_chain.run(question)
                        st.write(answer)
                    else:
                        st.warning("Please enter a question.")
            else:
                st.error("Failed to set up Question & Answer functionality.")
        
        # ========================
        # Example Usage
        # ========================
        elif st.session_state.analysis_type == "Generate Blog Post":
            blog_post = youtube_to_blog_post(text=st.session_state['summary'],model=st.session_state['model'],api_key=st.session_state['api_key'])
            if blog_post:
                st.markdown(blog_post)
    # ==========================
    # Error Handling
    # ==========================
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
