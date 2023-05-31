# Bring in deps
import os
from dotenv import load_dotenv

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

load_dotenv()

os.getenv('OPENAI_API_KEY')

# App framework 
st.title('TikTok GPT Creator')
prompt = st.text_input('Enter a prompt for the AI to complete:')

#Prompt Template
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = "write me a Tiktok title about {topic}"
)

script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'],
    template = "write me a Tiktok script based on this title TITLE: {title} while leverage this wikipedia research:{wikipedia_research}"
)


#memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

#create instance of OpenAI
llm = OpenAI(temperature=0.9) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)


wiki = WikipediaAPIWrapper()

#connect chains together. Uncomment this if don't use wikipediawrapper

# sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['title', 'script'], verbose=True)

#show stuff to the screen if there's a prompt 
if prompt: 
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.markdown(f'##### Title: {title}')
    st.markdown(f'##### Script: {script}')

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('Script History'): 
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)
    
