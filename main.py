import os
import validators
import streamlit as st
from urllib.parse import urlparse, parse_qs

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredURLLoader

from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
from youtube_transcript_api.proxies import WebshareProxyConfig
from youtube_transcript_api import YouTubeTranscriptApi
import dotenv
# Load environment variables from .env file
dotenv.load_dotenv()



os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")

# adding langsmith
os.environ["LANGSMITH_TRACING_V2"]=os.getenv("LANGSMITH_TRACING_V2", "true")
os.environ["LANGSMITH_TRACING"]=os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_ENDPOINT"]= os.getenv("LANGSMITH_ENDPOINT", "https://api.langsmith.com")
os.environ["LANGSMITH_API_KEY"]= os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"]="url summarizer"

# proxy
os.environ["WEBSHARE_PROXY_USERNAME"]= os.getenv("WEBSHARE_PROXY_USERNAME")
os.environ["WEBSHARE_PROXY_PASSWORD"]= os.getenv("WEBSHARE_PROXY_PASSWORD")


# ---------------------------- TRANSCRIPT LOADER ---------------------------- #
def get_transcript_from_url(youtube_url: str) -> str:
    try:
        # Setup proxy
        proxy_config = WebshareProxyConfig(
            proxy_username=os.environ["WEBSHARE_PROXY_USERNAME"],
            proxy_password=os.environ["WEBSHARE_PROXY_PASSWORD"],
        )

        parsed_url = urlparse(youtube_url)
        video_id = ""

        if "youtube.com" in youtube_url:
            video_id = parse_qs(parsed_url.query).get("v", [""])[0]
        elif "youtu.be" in youtube_url:
            video_id = parsed_url.path.lstrip("/")
        else:
            raise ValueError("URL does not appear to be a valid YouTube link.")

        if not video_id:
            raise ValueError("Could not extract video ID from URL.")

        # Try English first
        try:
            transcript = YouTubeTranscriptApi(proxy_config=proxy_config).get_transcript(video_id, languages=["en"])
        except NoTranscriptFound:
            # Fallback to Hindi or any available transcript
            transcript = YouTubeTranscriptApi(proxy_config=proxy_config).get_transcript(video_id, languages=["hi"])

        text = " ".join([t["text"] for t in transcript])
        return text

    except NoTranscriptFound:
        raise RuntimeError("No transcript available in English or Hindi.")
    except TranscriptsDisabled:
        raise RuntimeError("Transcripts are disabled for this video.")
    except Exception as e:
        raise RuntimeError(f"Failed to get transcript: {str(e)}")


# ---------------------------- STREAMLIT UI ---------------------------- #

st.set_page_config(page_title="LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

# Set default model
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-4.1"

st.session_state.selected_model = st.sidebar.selectbox(
    "Choose a model",
    ("gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini")
)
st.sidebar.info("If you hit a rate limit, try switching to another model.")
st.sidebar.markdown(f"You are using _{st.session_state.selected_model}_ by _OpenAI_")

# Load LLM if not yet or if model changed
if "llm" not in st.session_state or st.session_state.llm.model_name != f"openai/{st.session_state.selected_model}":
    st.session_state.llm = ChatOpenAI(model=f"openai/{st.session_state.selected_model}", temperature=0, streaming=True)

# Prompt Input
generic_url = st.text_input("Enter YouTube or Website URL", label_visibility="collapsed")

# Prompt Template
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Summarization Logic
if st.button("Summarize the Content from YT or Website"):
    if not validators.url(generic_url):
        st.error("❌ Invalid URL. Please provide a YouTube or Website link.")
    else:
        try:
            with st.spinner("⏳ Summarizing content..."):
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    transcript_text = get_transcript_from_url(generic_url)
                    docs = [Document(page_content=transcript_text, metadata={"source": generic_url})]
                    # st.write(docs)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
                        }
                    )
                    docs = loader.load()
                    # st.write(docs)

                chain = load_summarize_chain(
                    llm=st.session_state.llm,
                    chain_type="stuff",
                    prompt=prompt
                )

                summary = chain.invoke(docs)
                st.success("✅ Summary generated below:")
                st.write(summary["output_text"])

        except Exception as e:
            st.exception(f"❌ Exception: {e}")
