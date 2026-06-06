---
title: langgraph的一些经验
mathjax: true
date: 2026-06-06 16:20:23
tags:
  - Agent
---

## debug的一些技巧
使用时间旅行来进行debug，可以加快开发速度以及节约开发时的token调用

1. 通常我们可以把当前节点前的状态保存下来，然后我们使用fork，如下，之后会从`generate_ppt_content_per_page`节点的**后续节点**开始运行，当前的state就是你传入的state

```python
    fork_config = await agent.aupdate_state(
        config, origin_state, as_node="generate_ppt_content_per_page"
    )
    # 正常调用就是，input=None
     async for chunk in agent.astream(
            None,
            config=fork_config,
            durability="sync",
            stream_mode=["custom", "values", "tasks", "updates"],
            version="v2",
        ):
```

<!-- more -->

```python
    agent = await PPTAgent.create()
    thread_id = "zwc_test555"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    origin_state = State(
        ppt_info=PPTInfo(
            target_audience="导师以及同学",
            user_role="学生",
            layout_style="top_bottom",
            num_pages=10,
            theme="DeekSeek R1的介绍",
        ),
        messages=[],
        ppt_template_path="user_data/zwc_test/template/template.pdf",
        # current_timeline=TimeLine.INFO_GATHERED,
        have_ppt_content_files=False,
        # ppt_content_source_urls=["https://ghostty.org/docs/install/binary"],
        # ppt_content_files_markdown_contents=ppt_content_files_markdown_contents,
        have_ppt_template=False,
        web_fetch_results=web_fetch_results,
        ppt_outline=ppt_outline,
        user_content=user_content,
        ppt_page_contents=ppt_page_contents,
        # first_draft_results=first_draft_results,
        user_ppt_style="绿色简约风",
    )
    fork_config = await agent.aupdate_state(
        config, origin_state, as_node="generate_ppt_content_per_page"
    )
    async for chunk in agent.astream(
        None,
        config=fork_config,
        durability="sync",
        stream_mode=["custom", "values", "tasks", "updates"],
        version="v2",
    ):
        if chunk["type"] == "values" and len(chunk["interrupts"]) > 0:
            print({"type": "interrupts", "data": chunk["interrupts"]})
        if chunk["type"] == "custom" and "current_stage" in chunk["data"]:
            print({"type": "current_stage", "data": chunk["data"]["current_stage"]})
        # 初稿
        if chunk["type"] == "updates" and "generate_first_draft_task" in chunk["data"]:
            print(
                "updates",
                chunk["data"]["generate_first_draft_task"]["first_draft_results"],
            )
        # 终稿
        if chunk["type"] == "updates" and "generate_final_ppt_task" in chunk["data"]:
            print(
                "updates", chunk["data"]["generate_final_ppt_task"]["final_ppt_results"]
            )
```

2. 我们还可以使用checkpointer，将相关的信息持久化到磁盘，这样我们只需要运行了一次之后，状态会持久到磁盘，只要使用相同的thread_id，我们就可以找到任何一个节点前的状态。**开发阶段建议持久化设置为**`sync`**，来保证一定持久化到了磁盘，不然debug的时候可能有时候执行完上一个节点，我们在当前节点中止了程序，但是如果是**`async`**，可能checkpoint没有持久化到磁盘，**代码如下：

**第一次运行**

```python
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict, NotRequired
from langchain_core.utils.uuid import uuid7

from langgraph.checkpoint.sqlite import SqliteSaver


class State(TypedDict):
    topic: NotRequired[str]
    joke: NotRequired[str]


def generate_topic(state: State):
    return {"topic": "socks in the dryer"}


def write_joke(state: State):
    return {"joke": f"Why do {state['topic']} disappear? They elope!"}


with SqliteSaver.from_conn_string("test.db") as checkpointer:
    graph = (
        StateGraph(State)
    .add_node("generate_topic", generate_topic)
    .add_node("write_joke", write_joke)
    .add_edge(START, "generate_topic")
    .add_edge("generate_topic", "write_joke")
    .compile(checkpointer=checkpointer)
)
    thread_id = "123"
    # 第一次运行
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({}, config,durability="sync")

```

**第二次运行**，使用相同的thread__id，我们打算从`write_joke`节点开始运行，于是可以像下面这样找到对应的节点及状态

```python
 before_joke = next(s for s in history if s.next == ("write_joke",))
 fork_config = graph.update_state(
        before_joke.config,
        values=before_joke.values
    )

```

之后就可以像下面这样运行，`write_joke`之前的节点都不会运行，这样就可以快速地进行调试，避免每次调试都重复运行某些节点

```python
    fork_result = graph.invoke(None, fork_config)
    print(fork_result["joke"])
```



```python
with SqliteSaver.from_conn_string("test.db") as checkpointer:
    graph = (
        StateGraph(State)
    .add_node("generate_topic", generate_topic)
    .add_node("write_joke", write_joke)
    .add_edge(START, "generate_topic")
    .add_edge("generate_topic", "write_joke")
    .compile(checkpointer=checkpointer)
)
    thread_id = "123"
    config = {"configurable": {"thread_id": thread_id}}
    history = list(graph.get_state_history(config))
    before_joke = next(s for s in history if s.next == ("write_joke",))

    # Fork: update state to change the topic
    fork_config = graph.update_state(
        before_joke.config,
        values=before_joke.values
    )

    fork_result = graph.invoke(None, fork_config,durability="sync")
    print(fork_result["joke"])

```

## cache的技巧
```python
# cache的函数，通常包括对输入进行序列化处理，哪些字段不需要作为缓存的key
# 如果之后调用的时候输入的这些key相同就会命中缓存，直接返回结果
def generate_ppt_content_per_page_key_func(state):
    user_config = get_config()
    model_name = user_config["search_model_config"]["model"]
    include_fields = {
        "have_ppt_content_files",
        "user_content",
        "ppt_info",
        "ppt_outline",
    }
    # 如果是pydantic对象，就序列化为dict
    if isinstance(state, BaseModel):
        state = state.model_dump()
        
    state_dict = {k: v for k, v in state.items() if k in include_fields}
    state_dict["model_name"] = model_name
    return pickle.dumps(state_dict, protocol=5, fix_imports=False)

# 在构建graph设置cache_policy即可，可以设置缓存的时间
graph.add_node(
    "generate_ppt_content_per_page",
    generate_ppt_content_per_page,
    cache_policy=CachePolicy(
        key_func=generate_ppt_content_per_page_key_func,
        ttl=None if is_development() else 3600 * 24,
    ),
)


        
```

## 持久化的一些坑
langgraph默认的持久化模式为`async`，会在下一步运行的同时，异步进行持久化，如果运行中崩溃，可能某个检查点并没有写入到磁盘

`sync`为在下一个超级步开始执行进行持久化，会稍微影响性能

`exit`为只有在graph退出的时候（无论是退出时是否有错误）或者  human in the loop 中断触发时进行持久化

> 在开发的时候建议设置为sync，保证每次都持久化了，保证调试的时候不会出现state没有持久化到磁盘的情况，生产环境可以设置为async
>

![](https://img.leftover.cn/img-md/202606061623462.png)
