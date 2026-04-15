---
title: 使用openai sdk创建一个ReAct循环
mathjax: true
date: 2026-04-15 17:22:31
tags:
  - Agent
---

1. 为了测试，我们创建两个tool，一个是订机票的工具，一个是查询天气的工具

在创建工具的时候，我们需要编写函数的信息，如下所示,包括函数名，函数的描述，参数等信息，这些信息在调用模型时会随着message一起传给模型

```python
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海、广州",
                    }
                },
                "required": ["location"],
            },
        },
    },
```

手动写这些信息太麻烦了，我们可以借助sdk中的`pydantic_function_tool`函数，将`pydantic`对象解析为一个`function tool`，解析后的格式也和上面一样

<!-- more -->

2. ReAct循环的核心逻辑

初始化messages数组

```plain
1. 调用LLM，将LLM回复结果添加到messages数组中
2. 如果LLM需要调用工具，则我们根据LLM返回的工具调用的信息，根据参数信息调用对应的函数，得到工具调用的结果，然后把工具调用的结果加入到messages数组中，如果有多个工具调用，则多个工具调用的结果都需要加入,回到1
3. 如果LLM不需要调用工具，则将LLM的回复结果返回给用户，结束
```

3. 最终的代码

```python
import json
import os
import uuid

from dotenv import load_dotenv
from openai import OpenAI, pydantic_function_tool
from pydantic import BaseModel, Field

load_dotenv("test.env")


class FlightBooking(BaseModel):
    """当用户明确要求购买、预订机票，并且提供了必要的行程信息时，调用此工具生成机票订单。"""  # 这里对应工具的总体描述

    departure: str = Field(description="出发城市，例如：北京、新加坡")
    destination: str = Field(description="目的城市，例如：上海、东京")
    date: str = Field(description="起飞日期，必须符合 YYYY-MM-DD 格式")
    passenger_name: str = Field(description="乘机人的真实姓名")
    flight_class: str = Field(
        default="economy", description="舱位等级。可选：economy, business, first"
    )


def get_current_weather(location: str):
    """一个模拟查询天气的本地函数"""
    weather_info = {"location": location, "temperature": "22", "condition": "晴朗"}
    return json.dumps(weather_info)


def book_flight_ticket(
    departure: str,
    destination: str,
    date: str,
    passenger_name: str,
    flight_class: str = "economy",
):
    booking_id = f"FLIGHT-{str(uuid.uuid4())[:8].upper()}"
    mock_response = {
        "status": "success",
        "booking_id": booking_id,
        "passenger": passenger_name,
        "itinerary": f"{departure} 飞往 {destination}",
        "date": date,
        "cabin_class": flight_class,
        "message": "机票出票成功，已发送确认短信。",
    }

    return json.dumps(mock_response, ensure_ascii=False)


available_functions = {
    "get_current_weather": get_current_weather,
    "book_flight_ticket": book_flight_ticket,
}


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
)

messages = [
    {
        "role": "user",
        "content": "帮我看看今天北京的天气怎么样？如果天气不错的话，帮我预订一张2026年4月17日从上海飞往北京的经济舱机票，乘客名字是张三。",
    }
]


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海、广州",
                    }
                },
                "required": ["location"],
            },
        },
    },
    pydantic_function_tool(FlightBooking, name="book_flight_ticket"),  # name表示函数名
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "book_flight_ticket",
    #         "description": "当用户明确要求购买、预订机票，并且提供了必要的行程信息时，调用此工具生成机票订单。",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "departure": {
    #                     "type": "string",
    #                     "description": "出发城市，例如：北京、新加坡",
    #                 },
    #                 "destination": {
    #                     "type": "string",
    #                     "description": "目的城市，例如：上海、东京",
    #                 },
    #                 "date": {
    #                     "type": "string",
    #                     "description": "起飞日期，必须符合 YYYY-MM-DD 格式。如果用户说的是'明天'，请根据当前系统时间推算具体日期。",
    #                 },
    #                 "passenger_name": {
    #                     "type": "string",
    #                     "description": "乘机人的真实姓名",
    #                 },
    #                 "flight_class": {
    #                     "type": "string",
    #                     "enum": ["economy", "business", "first"],
    #                     "description": "舱位等级。默认为 economy (经济舱)。可选值包括：economy, business (公务舱), first (头等舱)",
    #                 },
    #             },
    #             # 声明哪些参数是必须的，缺失时大模型会主动追问用户
    #             "required": ["departure", "destination", "date", "passenger_name"],
    #         },
    #     },
    # },
]

model_name = "gpt-5.4"


while True:
    # 步骤1：调用LLM
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto 表示让模型自动决定是否调用工具
    )

    response_message = response.choices[0].message
    # 将LLM的回复添加到messages数组中
    messages.append(response_message)
    # 步骤2：如果有工具调用，则我们需要根据工具调用的信息调用函数得到结果，
    tool_calls = response_message.tool_calls
    if tool_calls is not None:
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]

            function_args = json.loads(tool_call.function.arguments)

            function_response = function_to_call(**function_args)
            # 将工具调用的结果添加到messages数组中
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": function_response,
                }
            )
    else:
        # 步骤3：如果不需要工具调用，则结束
        print("最终回答是：", response_message.content)
        break

```

