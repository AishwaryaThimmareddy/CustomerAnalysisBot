import os
from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
   
)
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import BM25Retriever
from llama_index.llms import OpenAI,LangChainLLM,HuggingFaceLLM,HuggingFaceInferenceAPI,TogetherLLM
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.postprocessor import SentenceTransformerRerank
from llama_index import QueryBundle
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.embeddings import HuggingFaceEmbedding
import streamlit as st
import time
import base64
from pathlib import Path
 
st.set_page_config(page_title="Customer Survey Analysis", page_icon="📖", layout="centered", initial_sidebar_state="auto", menu_items=None)
header = st.container()
header.title("Customer Survey Analysis 📖")
header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)
def load_css():
    with open(r"static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)
load_css()
div=f"""
    <div class="watermark"> <span class="img-txt">Powered by</span> <img src="app/static/logo-1.png" width=32 height=32></div>
    """
st.sidebar.markdown(div,unsafe_allow_html=True)
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
st.cache_resource()
def img_to_bytes(img_path):
     img_bytes = Path(img_path).read_bytes()
     encoded = base64.b64encode(img_bytes).decode()
     return encoded
image_path = "Promptora.png"
with open(image_path, "rb") as f:
    image_bytes = f.read()
 
# Convert image to base64
image_base64 = base64.b64encode(image_bytes).decode()
 
# Embed HTML to position the log and display an even smaller image in the top-right corner

st.markdown(f'''
<style>
.st-emotion-cache-1avcm0n{{
    visibility: hidden
}}
</style>
''',unsafe_allow_html=True)
 
st.markdown(
            """
            <style>
                .st-emotion-cache-vj1c9o {
                    background-color:rgb(38 39 48 / 0%);
                }
            </style>
            """,
            unsafe_allow_html=True
        )

 
### Custom CSS for the sticky header
st.markdown(
    """
<style>
        .st-emotion-cache-vj1c9o {
            background-color:rgb(38 39 48 / 0%);
        }
    div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
        position: sticky;
        top: 0;
        background-color: rgba(38, 39, 48, 1);
        z-index: 999;
        text-align:center;
 
    .fixed-header {
        border-bottom: 0;
    }
</style>
    """,
    unsafe_allow_html=True
)
@st.cache_resource()
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
 

 
@st.cache_resource()
def initialize_rag():
    """
    Initializes the AdvancedRAG object and caches results.
    """
   
    class AdvancedRAG:
        def __init__(self):
            _ = st.secrets["OPENAI_API_KEY"]
            # load documents
            self.documents = SimpleDirectoryReader(r"test", required_exts=['.pdf']).load_data()
   
            # global variables used later in code after initialization
            self.retriever = None
            self.reranker = None
            self.query_engine = None
   
            self.bootstrap()
        # initialize LLMs in below i provided different LLm Methods you can selectras your choice
        def bootstrap(self):
       
            llm = OpenAI(model="gpt-4",
                        api_key=st.secrets["OPENAI_API_KEY"],
                        temperature=0.1,system_prompt="You have a expertise knowledge in understanding the documents provided and after understanding thoroughly you are able to predict the future outcomes based on the past outcomes and on the present data provided.Your Future outcome predictions should be valid and accurate.Don't provide the false response. ")
               
           
       
            # initialize service context (set chunk size)
            service_context = ServiceContext.from_defaults(chunk_size=1024, llm=llm,embed_model=HuggingFaceEmbedding())
            nodes = service_context.node_parser.get_nodes_from_documents(self.documents)
   
            # initialize storage context (by default it's in-memory)
            storage_context = StorageContext.from_defaults()
            storage_context.docstore.add_documents(nodes)
   
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                service_context=service_context,
            )
       
            # We can pass in the index, doctore, or list of nodes to create the retriever
            self.retriever = BM25Retriever.from_defaults(similarity_top_k=2, index=index)
   
            # reranker setup & initialization
            self.reranker = SentenceTransformerRerank(top_n = 2, model = "BAAI/bge-reranker-base")
       
            self.query_engine = RetrieverQueryEngine.from_args(
                retriever=self.retriever,
                node_postprocessors=[self.reranker],
                service_context=service_context,
            )
   
        def query(self, query):
            # will retrieve context from specific companies
            nodes = self.retriever.retrieve(query)
            reranked_nodes = self.reranker.postprocess_nodes(
                nodes,
                query_bundle=QueryBundle(query_str=query)
            )
            response = self.query_engine.query(str_or_query_bundle=query)
            return response
    adv_rag = AdvancedRAG()
    return adv_rag
if __name__ == "__main__":
    adv_rag=initialize_rag()
 
    if "messages" not in st.session_state:
        st.session_state.messages = []
    clear_history = st.sidebar.button("Clear chat history")
    if clear_history:
        with st.spinner('Wait for it...'):
            st.session_state.messages = []
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
       
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # React to user input
    if prompt := st.chat_input("What is up?"):
        st.markdown(
            """
        <style>
            .stChatMessage {
                text-align: justify;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )
        start_time=time.time()
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            response = adv_rag.query(f"Echo: {prompt}")
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                    # st.write("Time consuming: {:.2f}s".format(time.time() - start_time))
    