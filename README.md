# Instagram Content Generator

This is a Streamlit web application that generates optimized Instagram content using AI based on your style guide and performance data.

## Setup Instructions

### 1. Virtual Environment
The project uses a virtual environment to manage dependencies. To activate it:

```bash
source venv/bin/activate
```

### 2. Required Files
Upload these files through the web interface:
- **Style Guide (.docx)** - Your content style guide document
- **Performance Data (.xlsx)** - Your Instagram performance data

### 3. OpenAI API Key
You need an OpenAI API key to use the generator. Enter it in the sidebar of the web app.

### 4. Running the Streamlit App
```bash
source venv/bin/activate
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

## Dependencies
All required packages are listed in `requirements.txt` and have been installed in the virtual environment:
- pandas
- openpyxl
- python-docx
- langchain
- langchain-openai
- langchain-community
- faiss-cpu
- openai
- streamlit

## Features
- 📱 **Web Interface**: Easy-to-use Streamlit web app
- 📊 **File Upload**: Upload your style guide and performance data
- 🎯 **Goal Optimization**: Choose to optimize for likes, comments, views, or engagement
- 📈 **Analytics**: View performance insights and data preview
- 🔄 **Real-time Generation**: Generate optimized content instantly
- 📋 **Copy-friendly Output**: Easy to copy generated scripts

## Troubleshooting
- If you get "pandas module not found", make sure you're using the virtual environment
- If you get "OpenAI API key" error, enter your API key in the sidebar
- If you see Streamlit warnings when running with `python main.py`, use `streamlit run main.py` instead
- Make sure to upload at least one file (style guide or performance data) to generate content
