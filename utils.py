import shutil
import streamlit as st
import openai
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
import re
import os
import uuid

# ==========================
# Utility Functions
# ==========================

def get_model(api_key):
    """
    Initializes and returns the OpenAI model client.
    """
    base_url = st.secrets['base_url']
    return openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

def get_transcripts(url):
    """
    Fetches the transcript of a YouTube video given its URL.
    """
    print('Gathering transcript...')
    video_id = extract_video_id(url)
    return fetch_transcript(video_id=video_id)

def extract_video_id(youtube_url):
    """
    Extracts the YouTube video ID from various possible URL formats.
    """
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&]+)',  # Standard YouTube URL
        r'(?:https?:\/\/)?youtu\.be\/([^?&]+)',                      # Shortened YouTube URL
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^?&]+)'   # Embed URL
    ]
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    raise ValueError("Invalid YouTube URL format. Please check the URL.")

def fetch_transcript(video_id):
    """
    Fetches the transcript text and entries for a given YouTube video ID.
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = ' '.join([entry['text'] for entry in transcript])
        return full_text, transcript
    except Exception as e:
        st.error(f"Could not fetch transcript: {str(e)}")
        return None, None

def split_text(text, chunk_size=500, chunk_overlap=100):
    """
    Splits large text into smaller, overlapping chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

# ==========================
# Main Functionalities
# ==========================

def generate_summary(text, api_key, model):
    """
    Generates a summary for the provided text. Handles large texts by splitting them into smaller chunks.
    """
    print("Generating summary of the text.")
    
    # Recursive summary for texts longer than 20,000 characters
    if len(text) > 20000:
        summary = ''
        chunks = split_text(text=text, chunk_overlap=200, chunk_size=20000)
        for chunk in chunks:
            try:
                client = get_model(api_key=api_key)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes content concisely."},
                        {"role": "user", "content": f"Provide a comprehensive summary of the following text:\n\n{chunk}"}
                    ]
                )
                summary += response.choices[0].message.content
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")
                return None
        # Recursively summarize the combined summaries
        return generate_summary(text=summary, api_key=api_key, model=model)
    else:
        try:
            client = get_model(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes content concisely."},
                    {"role": "user", "content": f"Provide a comprehensive summary of the following text:\n\n{text}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating summary: {str(e)}")
            return None

def generate_quiz(text, api_key, model, num_questions=5):
    """
    Generates a quiz with multiple-choice questions based on the provided text.
    Handles long texts by processing them in chunks.
    """
    print("Generating a quiz based on the video transcript.")

    if len(text) > 20000:
        quiz = ''
        chunks = split_text(text=text, chunk_size=20000, chunk_overlap=200)
        for chunk in chunks:
            try:
                client = get_model(api_key=api_key)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert quiz generator who creates multiple-choice questions. Use the source provided to you only."},
                        {"role": "user", "content": f"Generate {num_questions} multiple-choice questions with 4 options each based on this text. Ensure a mix of difficulty levels.\n\n{chunk}"}
                    ]
                )
                quiz += response.choices[0].message.content
            except Exception as e:
                st.error(f"Error generating quiz: {str(e)}")
                return None
        return quiz
    else:
        try:
            client = get_model(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert quiz generator who creates multiple-choice questions. Use the source provided to you only."},
                    {"role": "user", "content": f"Generate {num_questions} multiple-choice questions with 4 options each based on this text. Ensure a mix of difficulty levels.\n\n{text}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating quiz: {str(e)}")
            return None
        
def create_unique_persist_directory(base_directory="./chroma_db"):
    unique_dir = os.path.join(base_directory, str(uuid.uuid4()))
    os.makedirs(unique_dir, exist_ok=True)
    return unique_dir

def create_vector_store(text_chunks):
    """
    Creates a vector store for embedding and similarity search using Chroma.
    """
    print("Creating vector store...")
    api_key = st.secrets["api_key"]
    dir=create_unique_persist_directory()
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = Chroma.from_texts(
            texts=text_chunks, 
            embedding=embeddings, 
            persist_directory=dir
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error setting up vector store: {str(e)}")
        return None
    
def setup_rag(text_chunks, api_key, model):
    """
    Sets up a Retrieval-Augmented Generation (RAG) pipeline for querying based on a vector store.
    """
    try:
        if 'vectorstore' not in st.session_state or st.session_state['vectorstore'] is None:
            st.session_state['vectorstore'] = create_vector_store(text_chunks=text_chunks)
        vectorstore = st.session_state['vectorstore']
        base_url = st.secrets['base_url']
        llm = ChatOpenAI(openai_api_key=api_key, model_name=model, base_url=base_url)
        qa_chain = RetrievalQA.from_chain_type(
            llm, 
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type="stuff"
        )
        return qa_chain, vectorstore
    except Exception as e:
        st.error(f"Error setting up RAG: {str(e)}")
        return None, None

# ==========================
# Advanced Processing
# ==========================

def split_transcript_by_time(transcript, chunk_duration=300):
    """
    Splits a transcript into chunks based on a specified time interval (default: 5 minutes).
    """
    chunks = []
    current_chunk = []
    current_time = 0

    for entry in transcript:
        start_time = entry["start"]  # Timestamp when this text starts
        if start_time >= current_time + chunk_duration:
            # Close the current chunk and start a new one
            chunks.append(current_chunk)
            current_chunk = []
            current_time += chunk_duration

        current_chunk.append(entry)

    # Add any remaining entries to the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# Generalized system prompt
system_prompt = """
    You are an expert in processing long-form transcripts to identify engaging, meaningful short-form content. Your task is to extract a cohesive section from the transcript that meets these requirements:

You may combine content from multiple consecutive segments of the transcript.
The final output should flow logically without skipping or rearranging the order of the original content.
The output length should vary between 15 and 30 seconds when read aloud, ensuring flexibility based on the content.
Do not rephrase or modify the original transcript text. Use the provided wording exactly as is.
Ensure the output is clear, contextually complete, and can stand alone as a meaningful short-form clip.
For each selected segment, provide:

The start time of the segment.
The exact text as-is from the transcript.
A brief explanation of why the segment is engaging.


"""

def find_short_form_timestamps(transcript, api_key, model):
    """
    Identifies engaging short-form content timestamps within a video transcript.
    """
    print("Finding timestamps suitable for short-form content.")
    interesting_segments = []
    text_chunks = split_transcript_by_time(transcript=transcript)

    for chunk in text_chunks:
        try:
            client = get_model(api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""
Using the provided transcript of a video, extract segments of 10–30 seconds each to create short-form content. 
Transcript: {chunk}.
If no such segments exist, output: 'No such segment.'
"""}
                ]
            )
            interesting_segments += response.choices[0].message.content
            st.write(response.choices[0].message.content)
            st.write('---')
        except Exception as e:
            st.error(f"Error finding short-form timestamps: {str(e)}")
            return None
        
def youtube_to_blog_post(text, api_key, model, blog_title=None):
    
    try:
        summary = generate_summary(text, api_key, model)
        if not summary:
            st.error("Unable to generate a summary for the transcript.")
            return None

        
        if not blog_title:
            st.info("Generating a blog title.")
            try:
                client = get_model(api_key=api_key)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a creative assistant who generates catchy blog titles."},
                        {"role": "user", "content": f"Generate a catchy blog title for the following content:\n\n{summary}"}
                    ]
                )
                blog_title = response.choices[0].message.content.strip()
            except Exception as e:
                st.error(f"Error generating blog title: {str(e)}")
                return None

        st.info("Formatting the blog post.")
        blog_post = f"## {blog_title}\n------\n{summary}\n\n"
        blog_post += "### Key Takeaways:\n\n"
        
        # Optionally extract key points
        try:
            client = get_model(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant who summarizes content into key bullet points."},
                    {"role": "user", "content": f"Summarize the following content into key takeaways:\n\n{summary}"}
                ]
            )
            key_points = response.choices[0].message.content.strip()
            blog_post += key_points
        except Exception as e:
            st.warning(f"Could not generate key takeaways: {str(e)}")

        st.success("Blog post generated successfully.")
        return blog_post

    except Exception as e:
        st.error(f"Error creating blog post: {str(e)}")
        return None



