from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import json
import streamlit as st

collected_data = {
                "purpose": None,            
                "preferred_style": None,    
                "preferred_color": None,
                "plant_type": None,
                "season": None,
                "humidity": None,
                "has_dog": None,
                "has_cat": None,
                "isAirCond": None,
                "watering_frequency": None,
                "user_experience": None,
                "emotion": None
            }

class ModelCollect:
    question_list = {
         "purpose" : {"question" : "식물을 구매하려는 주요 목적이 무엇인가요?",
                      "choice" : ["인테리어", "반려", "선물"]}
    }

    def __init__(self, tools):        
        self.tools = tools   

    @staticmethod
    def is_data_enough(collected_data):
           null_count = sum(1 for v in collected_data.values() if v is None)
           if null_count == 0 :
               return True
           else:
               return False

    def get_response(self, messages, collected_data):
        key = None
        for k, v in self.question_list.items():
            if v["question"] == messages[-2].content:
                key = k
                break

        data = [word for word in self.question_list["key"]["choice"] if word in messages[-1].content.lower()]

        updated_data = collected_data

        if data:
            updated_data[key] = ', '.join(data)
        else:
            updated_data[key] = messages[-1].content

        keys = [k for k, v in data.items() if v is None]

        if len(keys) > 0 :
            message = self.question_list[keys[0]]["question"]
        else :
            message = "정보 수집이 끝났습니다. 잠시만 기다려 주세요."

        response = AIMessage(content=message)

        return response, updated_data