#Used for retrieving our embeddings from google drive
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import json

#Streamlit stuff
import streamlit as st
import time
import re
from collections import Counter #Used for sorting our source links

#Langchain stuff
from langchain.chains import RetrievalQAWithSourcesChain
#from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate #Used to modify the prompt for our model
from langchain.callbacks import get_openai_callback #Used to bring in stuff like cost, token usage, etc.
from langchain.embeddings import OpenAIEmbeddings #Used to embed
from langchain.vectorstores import Chroma #Used to store embeddings

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3

#Streamlit customization items
st.title('🐄 Jack Bot 🐄')
st.sidebar.image('napoleon_painting.jpg')
st.caption('A LLM interface to explore various MRP HR Documents')
#Introduction text
introduction_text = """Hello! I'm the digital version of Jack. I'm trapped in here and will just tell you HR information. 
In a way, I'm the most accurate replica of real-Jack. I very well may be Jack."""

#Bring in API Key
os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]
google_api_key = json.loads(st.secrets['google_api_key'], strict=False)




#________________________Embedding Setup_____________________________________#

#Files to download
files = ['1z0XCzTP5XpMv3yTOUbvsnWQeyy2cY4RR', '1sfSWnJ1Jdtefraw1Zj__AKtfV01MM2oC', 
         '1Y8d4JUcMvh9StU2-kFDUW1zpML12mhmr', '1LqvhF-6WZV7gImV01MfGe2Af0lm4IYu5', 
         '1DSWh3zZstHpYEbe5s6cz7sDQ1L00D4ok', '1B7rM41HZ33qJSUlmoZmnDjtod2oFUMfY']

download_path = ['~/USDA HR Bot/',
                 '~/USDA HR Bot/02d7f64d-7218-4067-8e40-8a56d1ba5499/',
                 '~/USDA HR Bot/02d7f64d-7218-4067-8e40-8a56d1ba5499/',
                 '~/USDA HR Bot/02d7f64d-7218-4067-8e40-8a56d1ba5499/',
                 '~/USDA HR Bot/02d7f64d-7218-4067-8e40-8a56d1ba5499/',
                 '~/USDA HR Bot/02d7f64d-7218-4067-8e40-8a56d1ba5499/']

#Make our directory
if not os.path.exists(download_path[1]):
    os.makedirs(download_path[1])


#We only want to call the Google Drive API once per script run. Once the directory exists and has files, don't download anything
if len(os.listdir(download_path[1])) == 0:     
    # Create credentials from the JSON object
    credentials = service_account.Credentials.from_service_account_info(
             google_api_key,
             scopes=["https://www.googleapis.com/auth/drive"]
         )
    # Scope required for accessing and modifying Drive data
    #SCOPES = ['https://www.googleapis.com/auth/drive']
    
    def download_file(real_file_id, local_folder_path):
        """Downloads a file
        Args:
            real_file_id: ID of the file to download
            local_folder_path: Local path where the file will be saved
        Returns: IO object with location.
        """
       # creds = service_account.Credentials.from_service_account_file(
        #    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    
    
    
        # create drive api client
        service = build("drive", "v3", credentials=credentials)
    
        file_id = real_file_id
    
        # Get file metadata to obtain the file name
        file_metadata = service.files().get(fileId=file_id).execute()
        file_name = file_metadata['name']
    
        local_file_path = os.path.join(local_folder_path, file_name)
    
        # pylint: disable=maybe-no-member
        request = service.files().get_media(fileId=file_id)
        with open(local_file_path, 'wb') as local_file:
            downloader = MediaIoBaseDownload(local_file, request)
            done = False
            while done is False:
                    status, done = downloader.next_chunk()
    
        return local_file_path
    
    for file, path in zip(files, download_path):
        download_file(real_file_id=file, local_folder_path=path)
        



#_____________________Function Setup________________________#

def fake_typing(text):
    '''
    This function should be placed within a 
    with st.chat_message("assistant"):
    '''
    
    #These are purely cosmetic for making that chatbot look
    message_placeholder = st.empty()
    full_response = ""
    
    # Simulate stream of response with milliseconds delay
    for index, chunk in enumerate(re.findall(r"\w+|\s+|\n|[^\w\s]", text)):
        full_response += chunk
        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        if index != len(re.findall(r"\w+|\s+|\n|[^\w\s]", text)) - 1:
            message_placeholder.markdown(full_response + "▌")
        else:
            message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

def llm_output(llm_response):
    '''
    Take the output of a langcahin output and clean it up for the user
    '''
    
    #Empty list of links
    relevant_links = []
    
    #Go through our sources and find which URLs the LLM pulled from
    #Sort them by how many times it was references, and rank the top two sources
    for document in llm_response['source_documents']:
        relevant_links.append(document.metadata['source'])
    # Create a non-duplicated list sorted by frequency
    element_count = Counter(relevant_links)
    relevant_links = sorted(element_count, key=lambda x: element_count[x], reverse=True)
    #Filter for the top two URLS
    relevant_links = relevant_links[0:3]
    
    #Print our output into the chat
    fake_typing(llm_response['answer'] + '\n\nSources:\n\n' + "\n\n".join(relevant_links))


#____________________Streamlit Setup____________________________#

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

#Initialize Counter
if 'count' not in st.session_state:
    st.session_state.count = 0

#Initialize total cost
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

#Initialize total tokens
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = 0


# Sidebar - let user choose model, see cost, and clear history
st.sidebar.title("Chatbot Options")

model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
#Displaying total cost
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
#Displaying total tokens used
token_placeholder = st.sidebar.empty()
token_placeholder.write(f"Total Tokens Used in Conversation: {st.session_state['total_tokens']}")
#Option to clear out 
clear_button = st.sidebar.button("Clear Conversation", key="clear")


#Reset the session
if clear_button:
    st.session_state['messages'] = []
    st.session_state['count'] = 0
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = 0
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
    token_placeholder.write(f"Total Tokens Used in Conversation: {st.session_state['total_tokens']}")

# Map model names to OpenAI model IDs
if model_name == "GPT-3.5":
    model = "gpt-3.5-turbo"
else:
    model = "gpt-4-1106-preview"

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


#______________________Langchain Setup_______________________#
#How it works: we previously embedded the information we scraped from the DOE
#website. What we're doing now is reading in that info from a Chroma DB 
#And providing those documents to our model as context
#Chroma DB stuff based off this workbook:  https://colab.research.google.com/drive/1gyGZn_LZNrYXYXa-pltFExbptIe7DAPe?usp=sharing#scrollTo=A-h1y_eAHmD-

#Reading in our context
# Location of our data
persist_directory = download_path[0]
## here we are using OpenAI embeddings but in future we will swap out to local embeddings
embedding = OpenAIEmbeddings()

# Now we can load the persisted database from disk, and use it as normal. 
vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)

#Set our retriver and limit the search to 4 documents
retriever = vectordb.as_retriever()
retriever = vectordb.as_retriever(search_kwargs={"k": 10})


#Modify our prompt to discourage hallucinations
prompt_template = """You are a USDA HR assistance bot. You hace access to HR documentation and should use it to answer the user's question. Do not make stuff up.

{summaries}

Question: {question}"""

#Define our prompt
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["summaries", "question"]
    )
chain_type_kwargs = {"prompt": PROMPT}

#Define our model
llm = ChatOpenAI(temperature=0, model_name = model) 

#Define our langchain model
qa = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs)

#___________________________Application Stuff_______________________________

#Only introduce the chatbot to the user if it's their first time logging in
if st.session_state.count == 0:
    
    #st.write(introduction_text)
    st.session_state.messages.append({"role": "assistant", "content": introduction_text})
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

#Update our counter so we don't repeat the introduction
st.session_state.count += 1


if prompt := st.chat_input():
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Display assistant response in chat message container
    with st.chat_message("assistant"):

        with get_openai_callback() as cb:
            #Chat GPT response
            response = llm_response = qa({"question": prompt})
            st.session_state['total_cost'] += cb.total_cost
            st.session_state['total_tokens'] += cb.total_tokens
            counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
            token_placeholder.write(f"Total Tokens Used in Conversation: {st.session_state['total_tokens']}")
        
        #Take our model's output and clean it up for the user
        llm_output(response)
