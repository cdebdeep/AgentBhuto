import streamlit as st
import time
import re

import pandas as pd
import numpy as np


import plotly.figure_factory as ff

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor


from MyAI import MyAI
from MyEDAHelper import MyEDAHelper

def fnc_graph(user_eda_column):    
    
    with st.container(border=True):
        st.scatter_chart(df, y = [user_eda_column])
        st.info('Scatter Plot', icon="✨")
    with st.container(border=True):
        st.plotly_chart(figure_or_data=ff.create_distplot([df[user_eda_column].dropna()], group_labels=[user_eda_column]))
        st.info('Hist Plot', icon="✨")
    return



with st.sidebar:
    st.title("💀 Hi, I am :blue[Agent BHUTO] ",)

    openai_api_key = st.text_input(
        "OpenAI API Key", key="langchain_search_api_key_openai", type="password"
    )
    if openai_api_key:
        myai:MyAI = MyAI(api_key=openai_api_key)
        if myai:
            try:
                myai.ValidateLLM()    
                st.write('You are using LLM: '+ myai.GetLLM().model_name)                
            except Exception as e:
                #st.write(e)
                st.error('Please enter a valid  OpenAI API key to continue.', icon="🚨")
        else:
            #st.write('Please add valid  OpenAI API key to continue.')=
            st.error('Please enter a valid  OpenAI API key to continue.', icon="🚨")
        

    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/cdebdeep/AgentBhuto.git)"
    "[![Open in GitHub Codespaces](https://ideal-guacamole-q47g9p6xrj24xgj.github.dev/)"

st.title("🔎 EDA - With your file")

uploaded_file = st.file_uploader("Upload your csv file", type=("csv"))

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.info('The total row & column count is:', icon="💁‍♀️")                            
    st.write(df.shape)
    st.dataframe(df)

if uploaded_file and openai_api_key:
    #myai:MyAI = MyAI(api_key=openai_api_key)
    agentexecutor:AgentExecutor = myai.GetAgentExecutor(df=df)
    

tab1, tab2, tab3, tab4 = st.tabs(["QA", "EDA","EDA-Graph", "Generate"])

with tab1:
        st.header("Question and Answer")
        question1 = st.text_input(
        "Ask something about the file",
        placeholder="Ask question like What is the avarage value of your numeric column?",
        disabled=not uploaded_file,
        )
        if uploaded_file and question1 and  openai_api_key:
            if agentexecutor:
                try:
                    with st.spinner("Wait, response is generating... !"):
                        myedahelper:MyEDAHelper = MyEDAHelper(agentexecutor)
                        retval = myedahelper.fnc_qa(question1)                       
                        st.success(retval, icon="✅")
                except Exception as e:
                    st.error('No data to show !!.', icon="🚨")    
        else:
            st.warning("Please add your OpenAI API key & upload a file to continue.",icon="⚠️")
            

with tab2:
        st.header("EDA")
        question2 = st.text_input(
        "Insert data column name for eda",
        placeholder="Insert column name?",
        disabled=not uploaded_file,
        )
        if uploaded_file and question2 and  openai_api_key:
            if agentexecutor:
                try:
                    with st.container():
                            fnc_graph(question2)

                    with st.spinner("Wait, response is generating... !"):                  

                        myedahelper:MyEDAHelper = MyEDAHelper(agentexecutor)
                        retval = myedahelper.fnc_eda(question2)
                        message_placeholder = st.empty()                

                        # Simulate stream of response with milliseconds delay
                        full_response = ""
                        for chunk in re.split(r'(\s+)', retval):
                            full_response += chunk + " "
                            time.sleep(0.01)

                            # Add a blinking cursor to simulate typing
                            st.success('Here is the response:', icon="✅")
                            message_placeholder.markdown(full_response + "▌")

                            

                except Exception as e:
                    st.error('No data to show !!.', icon="🚨")
                
                    
        else:
            st.warning("Please add your OpenAI API key & upload a file to continue.",icon="⚠️")


with tab3:
        st.header("EDA-Graph")
        question3 = st.text_input(
        "Insert data column name for eda graph",
        placeholder="Insert column name?",
        disabled=not uploaded_file,
        )
        if uploaded_file and question3 and  openai_api_key:            
                try:            

                    with st.spinner("Wait, response is generating... !"):                  

                        fnc_graph(question3)                            

                except Exception as e:
                    st.error('No data to show !!.', icon="🚨")
                
                    
        else:
            st.warning("Please add your OpenAI API key & upload a file to continue.",icon="⚠️")            

with tab4:
        st.header("Generate")
        question4 = st.text_input(
        "Insert your query here",
        placeholder="where columnname is less than 10?",
        disabled=not uploaded_file,
        )
        if uploaded_file and question4 and  openai_api_key:
            if agentexecutor:
                try:
                    with st.spinner("Wait, response is generating... !"):
                        myedahelper:MyEDAHelper = MyEDAHelper(agentexecutor)
                        retval:pd.DataFrame = myedahelper.fnc_modifydata(question4)
                        if retval.items():
                            st.success('The total row & column count is:', icon="✅")                            
                            st.write(retval.shape)
                            st.write(retval)                        
                except Exception as e:
                    #st.error('No data to show !!.', icon="🚨")
                    st.error(e, icon="🚨")
        
        else:
            st.warning("Please add your OpenAI API key & upload a file to continue.",icon="⚠️")

   



st.write("Thanks for visiting!")



