# Instagram Content Generator

This is a Streamlit web application that generates optimized Instagram content using AI based on your style guide and performance data.

## Setup Instructions

### 1. Virtual Environment
The project uses a virtual environment to manage dependencies. To activate it:

```bash
source venv/bin/activate
```

### 2. Required Files
- **Performance Data**: `IGScraper.xlsx` (automatically loaded from your project directory)
- **Style Guide**: Optional - you can upload your own custom style guide (.docx) or use the default one

Make sure `IGScraper.xlsx` is in the same directory as `main.py`

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
- 📊 **Auto-loaded Performance Data**: Automatically loads your Instagram performance data
- 📝 **Flexible Style Guide**: Upload your own style guide or use the default one
- 🎯 **Goal Optimization**: Choose to optimize for likes, comments, views, or engagement
- 📈 **Analytics**: View performance insights and data preview
- 🔄 **Real-time Generation**: Generate optimized content instantly
- 📋 **Copy-friendly Output**: Easy to copy generated scripts
- 🎨 **Custom Queries**: Enter specific instructions for targeted content

## Troubleshooting
- If you get "pandas module not found", make sure you're using the virtual environment
- If you get "OpenAI API key" error, enter your API key in the sidebar
- If you see Streamlit warnings when running with `python main.py`, use `streamlit run main.py` instead
- If you get "file not found" errors, make sure the required files are in the same directory as `main.py`
- The app automatically loads your data files - no need to upload them each time
