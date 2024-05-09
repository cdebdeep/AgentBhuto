import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import scipy as sp
import re

from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.agents import AgentExecutor

from langchain_experimental.agents import create_pandas_dataframe_agent

class MyEDAHelper:
    def __init__(self, agent):
        self.agent = agent
        

    def fnc_qa(self, user_question):
        result = self.agent.run(user_question)    
        return result


    def fnc_eda(self,user_eda_column):
        summary_statistics = self.agent.run(f"What are the mean, median, mode, standard deviation, variance, range, quartiles, skewness and kurtosis of {user_eda_column}")
        
        normality = self.agent.run(f"Check for normality or specific distribution shapes of {user_eda_column}")
        
        outliers = self.agent.run(f"Assess the presence of outliers of {user_eda_column}")
        
        trends = self.agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_eda_column}")
        
        missing_values = self.agent.run(f"Determine the extent of missing values of {user_eda_column}")
        
        return   summary_statistics +"\n\n " + normality + "\n\n " + outliers + "\n\n " + trends + "\n\n " + missing_values


    def fnc_eda_missing_values(self,user_eda_column):
        missing_values = self.agent.run(f"Determine the extent of missing values of {user_eda_column}")
        return

    def fnc_eda_summary_statistics(self,user_eda_column):
        summary_statistics = self.agent.run(f"What are the mean, median, mode, standard deviation, variance, range, quartiles, skewness and kurtosis of {user_eda_column}")
        return summary_statistics

    def fnc_eda_normality(self,user_eda_column):
        normality = self.agent.run(f"Check for normality or specific distribution shapes of {user_eda_column}")
        return normality

    def fnc_eda_outliers(self,user_eda_column):
        outliers = self.agent.run(f"Assess the presence of outliers of {user_eda_column}")
        return outliers

    def fnc_eda_trends(self,user_eda_column):
        trends = self.agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_eda_column}")
        return trends
    
    def fnc_modifydataAsRaw(self,modify_query)->str:

        agent_executor:AgentExecutor=self.agent
        
        return agent_executor.run(modify_query)
    
    
    def fnc_modifydata(self,modify_query)->pd.DataFrame:

        agent_executor:AgentExecutor=self.agent
        str = agent_executor.run(modify_query)

        # Extract rows of data
        rowswithheader = re.findall(r'\|.*\|.*\|', str)
        #print(rowswithheader)
        rows = re.findall(r'\| *\d+ *\|.*\|', str)
        #print(rows)
        # Extract column headers from the first row
        columns = re.findall(r' *([^|]+) *\|', rowswithheader[0])
        #print(columns)
        # Prepare data for DataFrame
        data = []
        for row in rows[0:]:
            values = re.findall(r' *([^|]+) *\|', row)
            data.append(values)
        #print(data)
        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        return df
    

    
