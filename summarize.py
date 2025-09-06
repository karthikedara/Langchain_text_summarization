import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader

#Streamlit UI

st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

#Get API key & URL field
with st.sidebar:
    groq_api_key = st.text_input("Enter the API",value='',type='password')

url = st.text_input("URL",label_visibility="collapsed")

import os
os.environ["GROQ_API_KEY"] = groq_api_key
llm = ChatGroq(model="openai/gpt-oss-120b",groq_api_key=groq_api_key)
prompt = """
Summarize the below tet into a 300 words
content:{text}"""
template = PromptTemplate(input_variables=['text'],template=prompt)

if st.button("Summarize the content from YT and website"):
    if not groq_api_key.strip()  or not url.strip():
        st.error("please provide the URL and api key")
    elif not validators.url(url):
        st.error("please enter the valid URL")
    else:
        try:
            with st.spinner("Waiting..."):
                if "youtube.com" in url:
                    loader = YoutubeLoader.from_youtube_url(url,add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[url],ssl_verify=False)
                docs = loader.load()
                chain = load_summarize_chain(llm=llm,chain_type="stuff",prompt=template)
                out_summary = chain.run(docs)
                st.success(out_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")
