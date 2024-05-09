import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import scipy as sp
import os

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor

from langchain_experimental.agents import create_pandas_dataframe_agent

class MyAI:    
    def __init__(self, api_key):
        self.api_key = api_key
        self.llm = ChatOpenAI(temperature=0, openai_api_key=self.api_key)     


    def GetAgentExecutor(self,df:pd.DataFrame):
        agent_executer:AgentExecutor =create_pandas_dataframe_agent(self.llm, verbose=True,agent_type="openai-tools",agent_executor_kwargs={"handle_parsing_errors": True},df=df)
        return agent_executer
    
    def GetLLM(self):
        return self.llm
    
    def ValidateLLM(self):
        return self.llm.invoke('Hi')
        
    
    def GetAPIKey(self):
        return self.api_key
        
