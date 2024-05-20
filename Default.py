import streamlit as st
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import base64
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

st.set_page_config(page_title="Customer Survey Analysis", page_icon="📖", layout="centered", initial_sidebar_state="auto", menu_items=None)

@st.cache_resource()
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .main {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }


    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg(r'background2.jpg')
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

# Custom CSS for the sticky header
st.markdown(
    """
<style>
    div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
        position: sticky;
        top: 0;
        background-color: white;
        z-index: 999;
        text-align:center;

    .fixed-header {
        border-bottom: 0;
    }
</style>
    """,
    unsafe_allow_html=True
)
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


@st.cache_resource
# Function to get embeddings from file
def get_embeddings_from(embedding_path):
    embeddings = HuggingFaceEmbeddings()
    docsearch = FAISS.load_local(embedding_path, embeddings,allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0,api_key=st.secrets["OPENAI_API_KEY"])
    # llm = HuggingFaceEndpoint(
    #                 repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    #                 huggingfacehub_api_token=os.getenv("HUGGING_FACE_TOKEN"),
    #                 temperature=0.5,max_new_tokens=4096)
    chain = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type="stuff", return_source_documents=True, retriever=docsearch.as_retriever())
    return chain
    
# Main function
def main():
    chain = get_embeddings_from("HOMEINDEX")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    selected_question = st.sidebar.selectbox("Select a question:", (
        "What is the net confidence of all in personal financial position into 2024? ",
        "What is the Percentage difference between Confident and not confident in Personal Financial Position into 2024 for female?",
        "How much percentage of  stable strategists would not like to buy a new home in 2024?",
        "What is the Percentage difference between Confident and not confident in Personal Financial Position into 2024 for Stable Strategists?"
        
    ))
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
    if st.sidebar.button("Submit"):
        with st.chat_message("user"):
            st.markdown(selected_question)
    # Add user message to chat history
        with st.spinner("Thinking..."):
            st.session_state.messages.append({"role": "user", "content": selected_question})
            response = chain(f"{selected_question}")
            with st.chat_message("assistant"):
                st.markdown(response['answer'])
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        with st.spinner("Thinking..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = chain(f"Echo: {prompt}")
            with st.chat_message("assistant"):
                    st.markdown(response['answer'])
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})
    clear_history = st.sidebar.button("Clear chat history")
    if clear_history:
        st.session_state.messages = []
if __name__ == '__main__':
    main()