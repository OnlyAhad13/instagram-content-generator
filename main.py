import streamlit as st
import pandas as pd
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from docx import Document as DocxDocument
import re
import os
from io import BytesIO
import time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["LANGCHAIN_VERBOSE"] = "false"

# ----------------------
# Config / Defaults
# ----------------------
DEFAULT_STYLE_PATH = "June 2025- Scripts & Captions [Mirrored Aesthetics].docx"  # your uploaded doc
DEFAULT_DATA_PATH = "IGScraper.xlsx"  # your CSV/XLSX file path (existing in environment)

st.set_page_config(
    page_title="Instagram Content Generator",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.6rem; font-weight: 700; text-align: center; margin-bottom: 1rem; }
    .script-output { background-color: #f8f9fa; border-left: 4px solid #007bff; padding: 1rem; border-radius: 0.5rem; font-family: 'Courier New', monospace; margin: 1rem 0; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 0.5rem; color: white; text-align: center; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ----------------------
# Helper: load style doc + dataset
# ----------------------
@st.cache_data
def load_and_enhance_data(style_file_bytes=None):
    """
    Returns: style_docs (list of LangChain Documents), data_docs (list), df (pandas)
    style_file_bytes: bytes-like object read from uploaded file (or None to use default)
    """
    style_texts = []
    # style guide
    try:
        if style_file_bytes is not None:
            # read uploaded bytes
            doc = DocxDocument(BytesIO(style_file_bytes))
        else:
            if not os.path.exists(DEFAULT_STYLE_PATH):
                # return empty if default missing
                return [], [], None
            doc = DocxDocument(DEFAULT_STYLE_PATH)

        for i, p in enumerate(doc.paragraphs):
            text = p.text.strip()
            if text:
                # add small context indicator so retriever can surface sections
                context = f"[DocSec {i//10 + 1}] {text}"
                style_texts.append(context)
    except Exception as e:
        st.error(f"Error parsing style docx: {e}")
        return [], [], None

    style_docs = [Document(page_content=t, metadata={"source": "style_guide", "type": "expert_example", "relevance": "high"}) for t in style_texts]

    # dataset
    df = None
    data_docs = []
    if os.path.exists(DEFAULT_DATA_PATH):
        try:
            df = pd.read_excel(DEFAULT_DATA_PATH)
        except Exception:
            try:
                df = pd.read_csv(DEFAULT_DATA_PATH)
            except Exception:
                df = None
    else:
        # no file found — return style docs only
        return style_docs, [], None

    if df is None:
        return style_docs, [], None

    # normalize columns
    df.columns = df.columns.str.strip().str.lower()

    # ensure numeric metrics
    for col in ["likes", "comments", "views"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    # engagement metric
    df["engagement"] = df["likes"] * 0.4 + df["comments"] * 0.4 + df["views"] * 0.2
    try:
        df["performance_tier"] = pd.qcut(df["engagement"], q=3, labels=["low", "medium", "high"], duplicates="drop")
    except Exception:
        df["performance_tier"] = "unknown"

    # compose text from likely columns
    column_mappings = {
        "HOOK": ["hooks", "hook", "opening"],
        "BODY": ["description", "body", "content"],
        "CAPTION": ["captions", "caption", "copy"],
        "SUMMARY": ["summary", "notes", "key_points", "uncleaned ai response"]
    }

    def compose_row(row):
        parts = []
        for label, candidates in column_mappings.items():
            for c in candidates:
                if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
                    val = str(row[c]).strip()
                    if val.lower() != "nan":
                        parts.append(f"{label}: {val}")
                    break
        perf = row.get("performance_tier", "unknown")
        engagement = row.get("engagement", 0)
        parts.append(f"PERFORMANCE: {perf} tier ({int(engagement)} engagement score)")
        return "\n".join(parts)

    for _, r in df.iterrows():
        text = compose_row(r)
        if text.strip():
            data_docs.append(Document(page_content=text, metadata={
                "source": "performance_data",
                "likes": int(r.get("likes", 0)),
                "comments": int(r.get("comments", 0)),
                "views": int(r.get("views", 0)),
                "engagement": float(r.get("engagement", 0)),
                "performance_tier": str(r.get("performance_tier", "unknown")),
                "relevance": "high" if r.get("engagement", 0) > df["engagement"].quantile(0.7) else "medium"
            }))

    return style_docs, data_docs, df

# ----------------------
# Retriever creation (explicit API key passing)
# ----------------------
@st.cache_resource
def create_smart_retriever(docs, openai_api_key):
    if not docs:
        return None
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150, separators=["\n\n", "\n", ".", "!", "?", " ", ""], keep_separator=True)
        chunks = splitter.split_documents(docs)

        # explicitly pass API key to embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)
        vectorstore = FAISS.from_documents(chunks, embeddings)

        base_retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 8, "score_threshold": 0.65, "fetch_k": 16})
        # set up compressor with explicit key for the internal llm
        compressor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
        compressor = LLMChainExtractor.from_llm(compressor_llm)
        return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
    except Exception as e:
        st.error(f"Error creating retriever: {e}")
        return None

# ----------------------
# Goal-aware re-ranking wrapper
# ----------------------
def enhance_retrieval_for_goal(retriever, goal):
    def goal_aware_retrieval(query):
        # Use invoke method (preferred) with fallback
        docs = []
        if hasattr(retriever, "invoke"):
            try:
                docs = retriever.invoke(query)
            except Exception:
                docs = []
        elif hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(query)
        goal_enhanced = []
        for d in docs:
            meta = d.metadata or {}
            if meta.get("source") == "performance_data":
                score = meta.get(goal.lower(), 0) if isinstance(meta.get(goal.lower(), 0), (int, float)) else 0
                d.metadata["goal_relevance"] = score
                goal_enhanced.append(d)
            else:
                d.metadata["goal_relevance"] = 100
                goal_enhanced.append(d)
        goal_enhanced.sort(key=lambda x: x.metadata.get("goal_relevance", 0), reverse=True)
        return goal_enhanced[:6]
    return goal_aware_retrieval

# ----------------------
# Prompt according to June doc rules (strict)
# ----------------------
def create_strict_prompt():
    # This template enforces the exact medical spa format with detailed structure
    template = """
You are an expert Instagram content creator for medical spas and aesthetic clinics. Generate content in the EXACT format and style of the provided examples.

REQUIRED FORMAT STRUCTURE:

[HOOK - Specific camera direction and expression]
"[Hook text - 8 words or less, curiosity gap or contrarian]"

[BODY - Specific delivery style and movement]
"[Body content - storytelling, educational, or testimonial style with specific details, numbers, and emotional elements]"

[CTA - Specific action and camera direction]
"[Call-to-action with specific comment word or DM instruction, include location]"

FILMING NOTES: [Specific filming instructions, props, and delivery notes]

CONTEXT & EXAMPLES:
{context}

GOAL: Optimize for {goal} (likes / comments / views / engagement)

STYLE REQUIREMENTS:
- Use the EXACT format structure shown above
- Include specific camera directions in brackets [HOOK - Look directly at camera, slight smile]
- Include filming notes section
- Use emotional storytelling and specific details
- Include numbers, timeframes, and concrete results
- Make it feel personal and authentic
- Use conversational, expert tone
- Include location references when relevant
- Create urgency and exclusivity in CTAs

CONTENT GUIDELINES:
- Hook: Create curiosity gap, contrarian statement, or emotional hook
- Body: Mix storytelling with educational content, include specific details
- CTA: Direct action with specific comment word or DM instruction
- Filming Notes: Include camera directions, props, and delivery style

Generate ONE Instagram reel script in the EXACT format above, optimized for {goal}.
"""
    return PromptTemplate.from_template(template)

# ----------------------
# Post-processing enforcement
# ----------------------
def enforce_script_format(llm, prompt_text, max_retries=3):
    # Updated to match the new medical spa format
    required = ["[HOOK", "[BODY", "[CTA", "FILMING NOTES"]
    last_resp = ""
    
    st.write(f"🔍 **DEBUG: Starting format enforcement with {max_retries} retries**")
    
    for attempt in range(max_retries):
        try:
            st.write(f"🔄 **Attempt {attempt + 1}**: Calling LLM...")
            
            # Use invoke if available; else predict or call generate
            resp = None
            if hasattr(llm, "invoke"):
                st.write("📞 Using llm.invoke() method")
                resp = llm.invoke(prompt_text)
                if hasattr(resp, "content"):
                    text = resp.content
                    st.write(f"✅ Got response content: {len(text)} characters")
                else:
                    text = str(resp)
                    st.write(f"⚠️ Response has no content, using str(): {len(text)} characters")
            elif hasattr(llm, "predict"):
                st.write("📞 Using llm.predict() method")
                text = llm.predict(prompt_text)
                st.write(f"✅ Got response: {len(text)} characters")
            else:
                st.write("📞 Using llm.generate() fallback")
                # fallback: try .generate
                gen = llm.generate([{"role":"user","content":prompt_text}])
                text = gen.generations[0][0].text if gen.generations else ""
                st.write(f"✅ Got response: {len(text)} characters")
                
        except Exception as e:
            st.error(f"❌ LLM generation error on attempt {attempt + 1}: {e}")
            st.exception(e)  # Show full traceback
            return None

        last_resp = text.strip()
        st.write(f"📝 **Response preview**: {last_resp[:200]}...")
        
        # Check if all required sections are present
        missing_sections = [sec for sec in required if sec not in last_resp]
        if not missing_sections:
            st.success(f"✅ **All required sections found on attempt {attempt + 1}**")
            return last_resp
        else:
            st.warning(f"⚠️ **Missing sections on attempt {attempt + 1}**: {missing_sections}")
            if attempt < max_retries - 1:
                st.write("🔄 Retrying...")
                time.sleep(0.4)  # small delay before retry

    st.warning("❌ **Could not fully enforce format after all retries**")
    st.write(f"📝 **Returning last attempt**: {last_resp[:200]}...")
    return last_resp

# ----------------------
# Streamlit UI
# ----------------------
def main():
    st.markdown('<h1 class="main-header">📱 Instagram Content Generator (RAG)</h1>', unsafe_allow_html=True)
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Configuration")
        openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", help="Provide a standard API key (starts with sk-...)")
        st.markdown("---")
        st.subheader("🎨 Custom Query")
        custom_query = st.text_area("Custom Query (Optional)", placeholder="E.g., Create a fitness motivation script")
        st.markdown("---")
        st.subheader("🎯 Optimization Goal")
        goal = st.selectbox("Select your primary goal:", options=["likes", "comments", "views", "engagement"], index=1)
        st.markdown("---")
        st.subheader("📝 Style Guide (optional)")
        style_file = st.file_uploader("Upload custom style guide (.docx)", type=["docx"])
        if style_file is not None:
            st.success("Custom style guide uploaded")
        else:
            st.info(f"Using default style guide: {DEFAULT_STYLE_PATH}")

    if not openai_api_key:
        st.warning("Please add OpenAI API key in the sidebar to proceed.")
        return

    # warn if project key
    if openai_api_key.strip().startswith("sk-proj-"):
        st.warning("You provided a project key (sk-proj-...). Project keys may require additional project/org settings to work. If you see 401 errors, ask the client for a standard key (sk-...).")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("🚀 Generate Content")
        if st.button("🎬 Generate Instagram Script", type="primary"):
            with st.spinner("Analyzing data and generating script..."):
                # Step 1: load data
                try:
                    style_bytes = None
                    if style_file is not None:
                        style_file.seek(0)
                        style_bytes = style_file.read()
                    style_docs, data_docs, df = load_and_enhance_data(style_bytes)
                    all_docs = style_docs + data_docs

                    st.info(f"Loaded {len(style_docs)} style docs and {len(data_docs)} performance docs")
                    if not all_docs:
                        st.error("No docs available (style or data). Please upload files or ensure defaults exist.")
                        return
                except Exception as e:
                    st.error(f"Error loading data: {e}")
                    return

                # Step 2: build retriever (explicit key)
                retriever = create_smart_retriever(all_docs, openai_api_key)
                if retriever is None:
                    st.error("Retriever creation failed.")
                    return
                goal_retriever = enhance_retrieval_for_goal(retriever, goal)

                # Step 3: retrieval for context
                query = custom_query or f"Create high-{goal} Instagram reel script"
                try:
                    relevant_docs = goal_retriever(query)
                except Exception:
                    # fallback to direct retriever call
                    if hasattr(retriever, "invoke"):
                        try:
                            relevant_docs = retriever.invoke(query)
                        except Exception:
                            relevant_docs = []
                    elif hasattr(retriever, "get_relevant_documents"):
                        relevant_docs = retriever.get_relevant_documents(query)
                    else:
                        relevant_docs = []
                context = "\n\n---\n\n".join([d.page_content for d in relevant_docs]) if relevant_docs else ""

                # Step 4: generate with strict prompt
                prompt_template = create_strict_prompt()
                formatted_prompt = prompt_template.format(context=context, goal=goal)
                # include custom instruction as part of prompt if provided
                if custom_query:
                    formatted_prompt += f"\n\nCUSTOM FOCUS: {custom_query}"

                # instantiate LLM with explicit key
                llm = ChatOpenAI(model="gpt-4o", temperature=0.7, max_tokens=800, openai_api_key=openai_api_key)

                # Dynamic AI Processing Visualization
                st.markdown("### 🧠 **AI Brain Processing**")
                
                # Step 1: AI Thinking
                thinking_progress = st.progress(0)
                thinking_status = st.empty()
                
                thinking_status.text("🤔 AI is analyzing your request and context...")
                thinking_progress.progress(25)
                time.sleep(1)
                
                thinking_status.text("🔍 AI is understanding your optimization goal...")
                thinking_progress.progress(50)
                time.sleep(1)
                
                thinking_status.text("📚 AI is processing style guide and examples...")
                thinking_progress.progress(75)
                time.sleep(1)
                
                thinking_status.text("✨ AI is ready to create your script!")
                thinking_progress.progress(100)
                time.sleep(0.5)
                
                # Step 2: AI Generation
                st.markdown("### 🎬 **AI Creative Generation**")
                
                generation_progress = st.progress(0)
                generation_status = st.empty()
                
                generation_status.text("🎯 Crafting the perfect hook...")
                generation_progress.progress(20)
                time.sleep(1)
                
                generation_status.text("📝 Building engaging body content...")
                generation_progress.progress(40)
                time.sleep(1)
                
                generation_status.text("📢 Creating compelling call-to-action...")
                generation_progress.progress(60)
                time.sleep(1)
                
                generation_status.text("🎥 Adding filming directions...")
                generation_progress.progress(80)
                time.sleep(1)
                
                generation_status.text("✨ Finalizing your medical spa script...")
                generation_progress.progress(100)
                time.sleep(0.5)
                
                # Clear the status
                thinking_status.empty()
                generation_status.empty()
                
                # Generate the actual content
                result = enforce_script_format(llm, formatted_prompt, max_retries=3)

                # display
                if result:
                    st.success("✅ Script generated")
                    st.markdown("### 🏆 Your Medical Spa Instagram Script")
                    st.markdown("---")
                    
                    # Clean up the result formatting
                    cleaned_result = result
                    # Remove excessive equal signs
                    for i in range(20, 80):
                        cleaned_result = cleaned_result.replace("=" * i, "")
                    # Remove extra whitespace and clean up
                    cleaned_result = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_result.strip())
                    
                    # Display with better formatting
                    st.markdown(cleaned_result)
                    st.markdown("---")
                else:
                    st.error("❌ Failed to generate script. Check the debug info above.")

    with col2:
        st.header("ℹ️ How to Use")
        st.markdown("""
        **🎯 Getting Started:**
        1. Enter your OpenAI API key in the sidebar
        2. Add a custom query (optional) for specific content
        3. Choose your optimization goal
        4. Upload a style guide (optional)
        5. Click "Generate Instagram Script"
        
        **✨ Features:**
        - 🧠 AI-powered content generation
        - 📱 Medical spa focused scripts
        - 🎬 Professional filming directions
        - 📊 Goal-optimized content
        - 🔄 Dynamic processing visualization
        
        **💡 Tips:**
        - Be specific in your custom queries
        - Use style guides for consistent branding
        - Try different optimization goals for variety
        """)

    st.markdown("---")
    st.markdown("---", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
