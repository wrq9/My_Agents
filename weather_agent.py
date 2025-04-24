from openai import OpenAI
from dotenv import load_dotenv
import json
import requests
import pandas as pd
import streamlit as st

@st.cache_resource
def get_client():
    return OpenAI(
        base_url="https://api.deepseek.com/v1",
        api_key='...'
    )

client = get_client()

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，例如：深圳市、广州市、北京市"
                },
                "district": {
                    "type": "string",
                    "description": "区名称，例如南山区、天河区、海淀区。如果为空，则获取城市天气"
                },
                "type": {
                    "type": "string",
                    "description": "实时或预报"
                }
            },
            "required": ["city", "type", "district"],
            "additionalProperties": False
        }
    }
},
]

@st.cache_resource
def load_city_data():
    return pd.read_excel("AMap_adcode_citycode.xlsx")

messages = []
input_text = st.text_area("我是天气小助手，您需要查询哪个城市的天气？😊")
messages.append({"role": "user", "content": input_text})

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

if st.button("回复"):
    tool_call = response.choices[0].message.tool_calls[0] # 获取第一个工具调用
    tool_name = tool_call.function.name # 获取工具名称
    tool_args = json.loads(tool_call.function.arguments) # 获取工具参数


    def get_weather(city, district=None, type="实时"):
        # 读取xlsx文件
        df = load_city_data()
        citycode = df[df["中文名"] == city]["citycode"].values[0]
        if district:
            # 获取"中文名"为district且"citycode"为citycode的行
            adcode = df[(df["中文名"] == district) & (df["citycode"] == citycode)]["adcode"].values[0]
        else:
            adcode = city
        # 调用高德地图API获取天气信息
        api_key = "..."
        if type == "实时":
            extensions = "base"
        else:
            extensions = "all"
        url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={adcode}&key={api_key}&extensions={extensions}"
        response = requests.get(url)
        data = response.json()
        return data

    function_call_result = get_weather(**tool_args)
    messages.append(response.choices[0].message)
    messages.append({"role": "tool", "content": json.dumps(function_call_result), "tool_call_id": tool_call.id})

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
    )

    st.write(response.choices[0].message.content)
