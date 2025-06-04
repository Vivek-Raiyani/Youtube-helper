# YouTube Video Analysis Application

## Clone the Repository

```

cd Lightning_fast_ai_agent
```

## Install Dependencies
```
pip install -r requirements.txt
```

## Set Up API Keys
Create a .streamlit/secrets.toml file and add the following keys:
```
api_key = "your_openai_api_key" # for open ai embeddings
base_url = "your_openai_api_base_url"  # Optional: Sambanova base url
```
Run the Application
```
streamlit run app.py
```

## Usage
# Embed YouTube Video
Enter a YouTube video URL in the input box to display the video.
# Choose Analysis Type
Select one of the following options:

## Summarize:
Get a concise summary of the video.
## Generate Quiz:
Create a quiz from the transcript.
## Find Short-Form Timestamps:
Extract timestamps for short video segments.
## Blog Post Generation: 
Turn the video into a blog post.
## Interactive Q&A
Ask questions about the video transcript using advanced retrieval techniques.


Directory Structure

```
.
├── app.py                     # Main application file
├── utils.py                   # Utility functions for processing
├── requirements.txt           # Python dependencies
├── .streamlit/
│   └── secrets.toml           # API keys and secrets configuration
├── chroma_db/                 # Persistent storage for vector stores
├── README.md                  # Project documentation

```

