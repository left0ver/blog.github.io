---
title: FastAPI 异步回调
mathjax: true
date: 2026-09-22 01:09:03
tags:
  - python
  - Agent
---

# 什么是异步回调？
在FastAPI中和现代的后端开发中，异步回调也通常被称为webhooks。简单来说，传统的API调用是:

**客户端发送请求，然后服务端这边进行处理并返回结果。**

而异步回调则是：

**客户端发送请求的时候，附带提供一个callback_url，服务器在收到请求之后会立即响应，并在后台异步处理任务，等待任务完成后，服务器主动向客户端提供的那个URL发起请求，将结果放到请求体中。**

<!-- more -->

## 为什么要异步回调？

在目前的后端开发中，尤其是Agent开发，通常我们后端要处理长时间的任务，例如一个爬虫任务，或RAG中对输入的内容进行解析并入库、RAG在线阶段对用户问题进行改写等操作并召回相关片段最终回答用户的问题等，这些都是一个长时间的任务，客户端发起请求之后并不能立刻地得到答案，因此对于这类长时间的任务，我们通常是在服务端异步执行，那么前端怎么拿到最终的一个结果呢？这里通常有两种方法：

1. 前端请求之后，返回一个`task_id`，之后再根据`task_id`轮询调用查询结果的接口，查询任务的状态和结果。比较经典的一个例子是`mineru`的api，他们的API的实现就是返回`task_id`，然后我们根据`task_id`来查询任务的状态和结果
2. 前端请求的时候携带一个`callback_url`，服务端接收请求之后还是立刻返回`task_id` （或者返回任务已创建的msg），之后服务端在任务结果之后，主动向客户端提供的那个URL发起请求，将结果放到请求体中。

> 在实际的开发中，其实这两种都是需要实现的。通常请求体的参数中callback_url是可选的，如果提供了，我们就需要回调，没有则不需要回调。
>





## FastAPI实现异步回调
1. 注册回调接口的文档，这个router的作用主要就是来生成相关的回调接口的文档，`InvoiceEvent`就是发送给客户端的请求体，`{$request.body.callback_url}`指客户端传来的URL

```python
# callback_router.py
from fastapi import APIRouter
from pydantic import BaseModel

from .schema import InvoiceEvent

callback_router = APIRouter()



class CallbackResponse(BaseModel):
    """第三方接收到回调后，应该返回给你的数据结构"""
    ok: bool


@callback_router.post(
    "{$request.body.callback_url}", response_model=CallbackResponse
)
def invoice_notification(body: InvoiceEvent):
    pass
```

启动之后就能看到回调的一个接口文档了，也能看到我们回调的时候会传什么内容给前端

![](https://img.leftover.cn/img-md/202609220113237.png)

2. 实现接口

这里我们实现了具体的业务接口，用户调用`/invoice`接口的时候，我们会创建任务并异步执行，并直接将`task_id`等内容返回给前端。在任务结束之后，则会根据前端传入的`callback_url`进行回调

```python
async def long_time_task(event_id: str, task_id: str, callback_url: HttpUrl):
    """
    模拟一个耗时的任务，比如生成发票、保存数据库等
    """

    await asyncio.sleep(5)
    print("Long time task completed")
    callback_data = InvoiceEvent(
        event_id=event_id, invoice_id=task_id, status="created"
    )
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                str(callback_url), json=callback_data.model_dump()
            )
            print(response.json())
        except httpx.HTTPError as e:
            print(f"Failed to send callback: {e}")
```

```python
# openai_callback.py
import asyncio

import httpx
from fastapi import APIRouter, BackgroundTasks, FastAPI
from pydantic import BaseModel, HttpUrl

from .callback_router import callback_router
from .schema import InvoiceEvent

app = FastAPI()
router = APIRouter()
app.include_router(callback_router)
app.include_router(router)


# 1. 定义数据模型
class Invoice(BaseModel):
    """用户触发你 API 时传入的数据"""

    id: str
    customer_id: str
    amount: float
    callback_url: HttpUrl


async def long_time_task(event_id: str, task_id: str, callback_url: HttpUrl):
    """
    模拟一个耗时的任务，比如生成发票、保存数据库等
    """

    await asyncio.sleep(5)
    print("Long time task completed")
    callback_data = InvoiceEvent(
        event_id=event_id, invoice_id=task_id, status="created"
    )
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                str(callback_url), json=callback_data.model_dump()
            )
            print(response.json())
        except httpx.HTTPError as e:
            print(f"Failed to send callback: {e}")


@router.post("/invoice", callbacks=callback_router.routes)
async def create_invoice(invoice: Invoice, background_tasks: BackgroundTasks):
    """
    1. 用户触发你 API 时传入的数据
    2. 你在这里处理业务逻辑，比如创建发票、保存数据库等
    3. 然后你可以使用 BackgroundTasks 来异步发送回调请求给第三方
    """
    # 模拟生成一个事件 ID
    event_id = f"evt_{invoice.id}"
    task_id = f"task_{invoice.id}"

    background_tasks.add_task(
        long_time_task, event_id, invoice.id, invoice.callback_url
    )
    return {
        "message": "Invoice task submitted successfully",
        "event_id": event_id,
        task_id: task_id,
    }
    # print(f"callback_url: {invoice.callback_url}")
    # async with httpx.AsyncClient() as client:
    #     try:
    #         response = await client.post(str(invoice.callback_url), json=callback_data.model_dump())
    #         print(response.json())
    #         return {"message": "Invoice created successfully", "event_id": event_id}
    #     except httpx.HTTPError as e:
    #         return {"message": f"Failed to send callback: {e}"}

```

```python
"""schema.py"""
from pydantic import BaseModel


class InvoiceEvent(BaseModel):
    """你主动发送给第三方的回调数据结构"""

    event_id: str
    invoice_id: str
    status: str

```



client的接收回调的接口，接收到服务端的回调结果之后，返回ok

> 这里我们使用fastapi起一个接口来mock 客户端
>

```python
# external.py
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, HttpUrl

from .schema import InvoiceEvent

app = FastAPI()

@app.post("/callback/invoices")
def get_invoice_callback_result(invoice_result: InvoiceEvent):
    """
    处理回调请求的逻辑
    """
    print(f"Received callback for invoice {invoice_result.invoice_id} with status {invoice_result.status}")
    return {"ok": True}
```

### 启动：
客户端：

```python
uv run uvicorn external:app --host 127.0.0.1 --port 8040
```

服务端

```python
uv run uvicorn openai_callback:app --host 127.0.0.1 --port 8030  
```

之后我们向server端发送一个这样的请求,callback_url是`http://localhost:8040/callback/invoices`,服务端完成任务之后会进行回调

![](https://img.leftover.cn/img-md/202609220114434.png)

结果：

客户端的接口收到了服务端的回调并返回ok

```python
Received callback for invoice 1 with status created
INFO:     127.0.0.1:51801 - "POST /callback/invoices HTTP/1.1" 200 OK
```

服务端接收到了`/invoice`请求，完成了任务之后向`callback_url`回调，并收到了客户端返回的ok，确认客户端收到了回调的结果

```python
INFO:     127.0.0.1:51796 - "POST /invoice HTTP/1.1" 200 OK
Long time task completed
{'ok': True}
```

