import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st 
hf_token = st.secrets["HF_TOKEN"]
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
llmg=ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id='openai/gpt-oss-120b'))

documents = [
"Network issue.\n"
"Restart the router, check cables, or contact your internet provider."

"Email issue.\n"
"Check internet connection, verify email settings, and look in spam or junk folders."

"Password issue.\n"
"Use the “Forgot Password” option to reset and enable multi factor authentication."

"Storage issue.\n"
"Delete unnecessary files, clear temporary data, or move large files to cloud storage."

"Login issue.\n"
"Double check username and password, clear browser cache, and ensure the account is not locked."

"Software crash.\n"
"Update the software, reinstall if needed, and check compatibility with your system."

"Slow system performance.\n"
"Close unused applications, upgrade RAM if possible, and run antivirus scans."

"Printer issue.\n"
"Check connections, restart the printer, and reinstall drivers if printing fails."

"Update failure.\n"
"Ensure stable internet, free up disk space, and retry updates after restarting."

"Security issue.\n"
"Install antivirus software, keep systems patched, and avoid suspicious links or attachments."

"Browser issue.\n"
"Clear cache and cookies, update the browser, or try a different one."

"Mobile app issue.\n"
"Update the app, reinstall if needed, and check device compatibility."

"Connectivity issue.\n"
"Switch to a stable WiFi or mobile network and restart the device."

"File corruption issue.\n"
"Restore from backup or use repair tools to recover the file."

"Account lockout issue.\n"
"Contact support or use secure recovery options to regain access."

"Hardware overheating issue.\n"
"Clean dust from vents, improve airflow, or use cooling pads."

"Display issue.\n"
"Check cable connections, update drivers, or adjust display settings."

"Sound issue.\n"
"Check audio settings, update sound drivers, or reconnect speakers/headphones."

"Backup issue.\n"
"Verify backup settings, ensure enough storage, and test recovery regularly."

"Performance lag in applications.\n"
"Update the application, close background tasks, and allocate more system resources."
]
prompt=ChatPromptTemplate.from_template(
    """You are an assistant that answers questions  using the provided context.
    Context:{context}
    Question:{question}

    If the answer is not in the context, say:"Unable to help on the provided context"
    """

)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

documents_for_splitter=[Document(page_content=doc) for doc in documents]

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separators=['\n\n','\n'," ",""]

)
st.title("Smart AI Ticket Resoluter")
query=st.text_area("Enter you Problem")
chunks=text_splitter.split_documents(documents_for_splitter)
context1="\n".join(chunk.page_content for chunk in chunks)
chain=prompt|llmg
if st.button("Submit"):
    response1=chain.invoke({
    "context":context1,
    "question":{query}
})
    st.write(response1.content)


