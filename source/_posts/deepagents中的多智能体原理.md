---
title: deepagents中的多智能体原理
mathjax: true
date: 2026-06-01 22:56:17
tags:
  - Agent
---

# 任务分解和规划（TodoListMiddleware）
## 原理解析
在deepagents中，使用`create_deep_agent` api创建agent，会默认添加一个`write_todos`工具，可以让agent进行规划，其实现也很简单，也是依赖于langchain的中间件机制,langchain中提供了预构建的`TodoListMiddleware`

基本的原理在于：

1. 系统提示词中需要提示模型使用`write_todos`这个工具对复杂任务进行规划
2. 以及这个`write_todos`工具的描述，告诉模型什么时候用这个工具，什么时候不该用

<!-- more -->

下面是部分相关的代码，可以看到todo的schema中包含两个字段，任务的状态以及任务的描述

```python

class Todo(TypedDict):
    """A single todo item with content and status."""

    content: str
    """The content/description of the todo item."""

    status: Literal["pending", "in_progress", "completed"]
    """The current status of the todo item."""
@tool(description=WRITE_TODOS_TOOL_DESCRIPTION)
def write_todos(
    todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command[Any]:
    """Create and manage a structured task list for your current work session."""
    return Command(
        update={
            "todos": todos,
            "messages": [ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)],
        }
    )
```

该中间件核心的实现:

可以看到该中间件会在每次模型调用之前，添加有关todo的系统提示词，告知模型怎么使用`write_todos`这个工具。

```python
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Update the system message to include the todo system prompt.

        Args:
            request: Model request to execute (includes state and runtime).
            handler: Async callback that executes the model request and returns
                `ModelResponse`.

        Returns:
            The model call result.
        """
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return handler(request.override(system_message=new_system_message))
```

然后在模型调用介绍，会判断是否同时调用多次`write_todos`工具，如果是，则构建一条工具调用错误的msg，提示模型。

```python
    @override
    def after_model(
        self, state: PlanningState[ResponseT], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Check for parallel write_todos tool calls and return errors if detected.

        The todo list is designed to be updated at most once per model turn. Since
        the `write_todos` tool replaces the entire todo list with each call, making
        multiple parallel calls would create ambiguity about which update should take
        precedence. This method prevents such conflicts by rejecting any response that
        contains multiple write_todos tool calls.

        Args:
            state: The current agent state containing messages.
            runtime: The LangGraph runtime instance.

        Returns:
            A dict containing error ToolMessages for each write_todos call if multiple
            parallel calls are detected, otherwise None to allow normal execution.
        """
        messages = state["messages"]
        if not messages:
            return None

        last_ai_msg = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
        if not last_ai_msg or not last_ai_msg.tool_calls:
            return None

        # Count write_todos tool calls
        write_todos_calls = [tc for tc in last_ai_msg.tool_calls if tc["name"] == "write_todos"]

        if len(write_todos_calls) > 1:
            # Create error tool messages for all write_todos calls
            error_messages = [
                ToolMessage(
                    content=(
                        "Error: The `write_todos` tool should never be called multiple times "
                        "in parallel. Please call it only once per model invocation to update "
                        "the todo list."
                    ),
                    tool_call_id=tc["id"],
                    status="error",
                )
                for tc in write_todos_calls
            ]

            # Keep the tool calls in the AI message but return error messages
            # This follows the same pattern as HumanInTheLoopMiddleware
            return {"messages": error_messages}

        return None
```



当然，这是一种比较简单的任务分解，他将一个复杂的任务分解为多个顺序执行的小的任务。但通常会搭配subagent来一起使用

+ 如果你不搭配subagent来使用，基于ReAct这个范式下，通常会出现下面这种情况，首先它把任务进行了拆分，但是并没有分别地完成每个子任务，只是单纯地更新了任务状态，再在最后输出了最终的结果。

```python

================================ Human Message =================================

帮我调研 LangGraph 的技术架构，对比三个竞品，写一份 3000 字的分析报告。
================================== Ai Message ==================================
Tool Calls:
  write_todos (call_b5r3ATHhcvLLsNVu2ZyFDnsh)
 Call ID: call_b5r3ATHhcvLLsNVu2ZyFDnsh
  Args:
    todos: [{'content': '梳理 LangGraph 技术架构与核心机制', 'status': 'in_progress'}, {'content': '选择并分析三个可比竞品', 'status': 'pending'}, {'content': '形成对比维度并撰写约 3000 字中文报告', 'status': 'pending'}]
================================= Tool Message =================================
Name: write_todos

Updated todo list to [{'content': '梳理 LangGraph 技术架构与核心机制', 'status': 'in_progress'}, {'content': '选择并分析三个可比竞品', 'status': 'pending'}, {'content': '形成对比维度并撰写约 3000 字中文报告', 'status': 'pending'}]
================================== Ai Message ==================================
Tool Calls:
  write_todos (call_v5L519Py0WYNBpfhwGlOuitY)
 Call ID: call_v5L519Py0WYNBpfhwGlOuitY
  Args:
    todos: [{'content': '梳理 LangGraph 技术架构与核心机制', 'status': 'completed'}, {'content': '选择并分析三个可比竞品', 'status': 'in_progress'}, {'content': '形成对比维度并撰写约 3000 字中文报告', 'status': 'pending'}]
================================= Tool Message =================================
Name: write_todos

Updated todo list to [{'content': '梳理 LangGraph 技术架构与核心机制', 'status': 'completed'}, {'content': '选择并分析三个可比竞品', 'status': 'in_progress'}, {'content': '形成对比维度并撰写约 3000 字中文报告', 'status': 'pending'}]
================================== Ai Message ==================================
Tool Calls:
  write_todos (call_W81YMsDJCTLf3e5bJsyoHIoo)
 Call ID: call_W81YMsDJCTLf3e5bJsyoHIoo
  Args:
    todos: [{'content': '梳理 LangGraph 技术架构与核心机制', 'status': 'completed'}, {'content': '选择并分析三个可比竞品', 'status': 'completed'}, {'content': '形成对比维度并撰写约 3000 字中文报告', 'status': 'in_progress'}]
================================= Tool Message =================================
Name: write_todos

Updated todo list to [{'content': '梳理 LangGraph 技术架构与核心机制', 'status': 'completed'}, {'content': '选择并分析三个可比竞品', 'status': 'completed'}, {'content': '形成对比维度并撰写约 3000 字中文报告', 'status': 'in_progress'}]
================================== Ai Message ==================================
Tool Calls:
  write_todos (call_1yuxHsHjSpFjyhDFG1sI03fI)
 Call ID: call_1yuxHsHjSpFjyhDFG1sI03fI
  Args:
    todos: [{'content': '梳理 LangGraph 技术架构与核心机制', 'status': 'completed'}, {'content': '选择并分析三个可比竞品', 'status': 'completed'}, {'content': '形成对比维度并撰写约 3000 字中文报告', 'status': 'completed'}]
================================= Tool Message =================================
Name: write_todos

Updated todo list to [{'content': '梳理 LangGraph 技术架构与核心机制', 'status': 'completed'}, {'content': '选择并分析三个可比竞品', 'status': 'completed'}, {'content': '形成对比维度并撰写约 3000 字中文报告', 'status': 'completed'}]
================================== Ai Message ==================================

# LangGraph 技术架构与竞品对比分析报告
#...
```

+ 我们接下来搭配一个通用的subagent来使用,如果搭配subagent来使用的话，当我们把任务进行了拆分了之后，它会调用工具把任务分配给subagent，最后进行一个汇总

> 下面是最终的结果，可以看到agent调用了task这个工具（这是一个通用的subagent），等待subagent完成了之后继续完成接下来的任务
>

```python
================================ Human Message =================================

帮我调研 LangGraph 的技术架构，对比三个竞品，写一份 3000 字的分析报告。
================================== Ai Message ==================================
Tool Calls:
  write_todos (call_MrOleJOxx9ilhndDe9hfibpC)
 Call ID: call_MrOleJOxx9ilhndDe9hfibpC
  Args:
    todos: [{'content': '梳理报告结构与竞品选择标准', 'status': 'in_progress'}, {'content': '调研 LangGraph 技术架构要点', 'status': 'pending'}, {'content': '调研三个竞品并提炼对比维度', 'status': 'pending'}, {'content': '撰写约 3000 字中文分析报告', 'status': 'pending'}]
================================= Tool Message =================================
Name: write_todos

Updated todo list to [{'content': '梳理报告结构与竞品选择标准', 'status': 'in_progress'}, {'content': '调研 LangGraph 技术架构要点', 'status': 'pending'}, {'content': '调研三个竞品并提炼对比维度', 'status': 'pending'}, {'content': '撰写约 3000 字中文分析报告', 'status': 'pending'}]
================================== Ai Message ==================================
Tool Calls:
  write_todos (call_xiqmkMxWxRs6NSjnvK2tXJUI)
 Call ID: call_xiqmkMxWxRs6NSjnvK2tXJUI
  Args:
    todos: [{'content': '梳理报告结构与竞品选择标准', 'status': 'completed'}, {'content': '调研 LangGraph 技术架构要点', 'status': 'in_progress'}, {'content': '调研三个竞品并提炼对比维度', 'status': 'pending'}, {'content': '撰写约 3000 字中文分析报告', 'status': 'pending'}]
================================= Tool Message =================================
Name: write_todos

Updated todo list to [{'content': '梳理报告结构与竞品选择标准', 'status': 'completed'}, {'content': '调研 LangGraph 技术架构要点', 'status': 'in_progress'}, {'content': '调研三个竞品并提炼对比维度', 'status': 'pending'}, {'content': '撰写约 3000 字中文分析报告', 'status': 'pending'}]
================================== Ai Message ==================================
Tool Calls:
  task (call_8w0WB2nP1AAvCBlqVvfSnbH6)
 Call ID: call_8w0WB2nP1AAvCBlqVvfSnbH6
  Args:
    description: 请调研 LangGraph 的技术架构，聚焦截至 2024/2025 的公开信息。输出中文要点，包含：核心定位、Graph/State/Node/Edge/Conditional Edge、Pregel/BSP 执行模型或 super-step 思想、checkpoint/persistence、human-in-the-loop、memory/store、streaming、subgraph、LangGraph Platform/Studio/Cloud/Server（如有）、与 LangChain 关系、优劣势和典型场景。要求结构化、准确、可用于撰写 3000 字中文分析报告。
    subagent_type: general-purpose
  task (call_cnD2toMT7hjlubDmes6Jsdnf)
 Call ID: call_cnD2toMT7hjlubDmes6Jsdnf
  Args:
    description: 请调研 LangGraph 的三个竞品/替代方案并对比，建议选择 CrewAI、Microsoft AutoGen、LlamaIndex Workflows/Agents（如认为更合适可说明）。输出中文要点，包含每个产品的核心定位、架构抽象、编排方式、状态与持久化、人机协同、可观测/部署生态、适合场景、相对 LangGraph 的优势劣势。最后给一个对比表的文字版。要求结构化、准确、可用于撰写 3000 字中文分析报告。
    subagent_type: general-purpose
================================= Tool Message =================================
Name: task

以下为截至 2024/2025 年公开信息整理的 **LangGraph 技术架构调研要点**，可直接作为撰写约 3000 字中文分析报告的素材框架。
...

================================= Tool Message =================================
Name: task

以下调研以 **LangGraph** 作为参照对象，对三个常见竞品/替代方案进行对比：**CrewAI、Microsoft AutoGen、LlamaIndex Workflows/Agents**。三者都可用于构建 Agent 应用，但设计重心不同：CrewAI 更偏“角色化多智能体协作”，AutoGen 更偏“可编程的多智能体对话与实验框架”，LlamaIndex Workflows/Agents 更偏“围绕数据/RAG 的 Agent 与工作流编排”。如果目标是构建生产级、状态可控、可恢复的人机协同 Agent 系统，LangGraph 仍然是最强参照之一。
...
================================== Ai Message ==================================
Tool Calls:
  write_todos (call_W1Qy9kzPDkdv1nogvEu3MKPH)
 Call ID: call_W1Qy9kzPDkdv1nogvEu3MKPH
  Args:
    todos: [{'content': '梳理报告结构与竞品选择标准', 'status': 'completed'}, {'content': '调研 LangGraph 技术架构要点', 'status': 'completed'}, {'content': '调研三个竞品并提炼对比维度', 'status': 'in_progress'}, {'content': '撰写约 3000 字中文分析报告', 'status': 'pending'}]
================================= Tool Message =================================
Name: write_todos

Updated todo list to [{'content': '梳理报告结构与竞品选择标准', 'status': 'completed'}, {'content': '调研 LangGraph 技术架构要点', 'status': 'completed'}, {'content': '调研三个竞品并提炼对比维度', 'status': 'in_progress'}, {'content': '撰写约 3000 字中文分析报告', 'status': 'pending'}]
================================== Ai Message ==================================
Tool Calls:
  write_todos (call_YrzUrct9G6EmBUwsm7gbv3Ue)
 Call ID: call_YrzUrct9G6EmBUwsm7gbv3Ue
  Args:
    todos: [{'content': '梳理报告结构与竞品选择标准', 'status': 'completed'}, {'content': '调研 LangGraph 技术架构要点', 'status': 'completed'}, {'content': '调研三个竞品并提炼对比维度', 'status': 'completed'}, {'content': '撰写约 3000 字中文分析报告', 'status': 'in_progress'}]
================================= Tool Message =================================
Name: write_todos

Updated todo list to [{'content': '梳理报告结构与竞品选择标准', 'status': 'completed'}, {'content': '调研 LangGraph 技术架构要点', 'status': 'completed'}, {'content': '调研三个竞品并提炼对比维度', 'status': 'completed'}, {'content': '撰写约 3000 字中文分析报告', 'status': 'in_progress'}]
================================== Ai Message ==================================
Tool Calls:
  write_todos (call_pTlQIoj1EZbOOFeRJ7l8XeVM)
 Call ID: call_pTlQIoj1EZbOOFeRJ7l8XeVM
  Args:
    todos: [{'content': '梳理报告结构与竞品选择标准', 'status': 'completed'}, {'content': '调研 LangGraph 技术架构要点', 'status': 'completed'}, {'content': '调研三个竞品并提炼对比维度', 'status': 'completed'}, {'content': '撰写约 3000 字中文分析报告', 'status': 'completed'}]
================================= Tool Message =================================
Name: write_todos

Updated todo list to [{'content': '梳理报告结构与竞品选择标准', 'status': 'completed'}, {'content': '调研 LangGraph 技术架构要点', 'status': 'completed'}, {'content': '调研三个竞品并提炼对比维度', 'status': 'completed'}, {'content': '撰写约 3000 字中文分析报告', 'status': 'completed'}]
================================== Ai Message ==================================

# LangGraph 技术架构与竞品对比分析报告
...

```

## 局限性和改进
上面的todo list 是单纯地将一个复杂的任务拆分为多个顺序执行的小任务，对于大多数情况来说已经够了，但是任务之间没有明显的依赖关系，例如任务C必须等A和B 完成之后再执行。具体可看[Learn Claude Code 的s12](https://learn.shareai.run/zh/s12/)。思路差不多，我们需要修改todo item的schema，任务的状态需要添加`blockedBy`（即被谁阻塞），`blocks`(当前任务阻塞了哪些任务。)

我对官网预构建的`TodoListMiddleware`稍微改造了一下，具体的代码在[我的GitHub仓库](https://github.com/left0ver/study_agent/blob/main/study_deepagents/advance_todo_middleware.p)，让任务之间有依赖关系。

生成的`task_list`大概是这样的，如下所示，可以看到不同的任务之间有依赖关系，这个针对复杂的来说还是很有用的。当然我们还是可以添加更多的工具，让agent可以更加精细地管理task，目前的方式是直接让他更新对应的任务状态。这会需要模型比较强大，可能弱一点的模型对于任务的状态更新就会错乱、或者上下文长了之后模型对任务的更新就会不正确。之后我们可以提供一个任务更新的tool，传入task_id来更新任务。

```python
[
    {
        "content": "调研 LangGraph 技术架构、核心概念与适用场景",
        "depends_on": [],
        "id": "research_langgraph",
        "status": "in_progress",
    },
    {
        "content": "调研并选定三个竞品的技术架构与定位",
        "depends_on": [],
        "id": "research_competitors",
        "status": "in_progress",
    },
    {
        "content": "综合对比并撰写约 3000 字中文分析报告",
        "depends_on": ["research_langgraph", "research_competitors"],
        "id": "write_report",
        "status": "pending",
    },
]
```

全部的对话记录：

```python
================================ Human Message =================================

帮我调研 LangGraph 的技术架构，对比三个竞品，写一份 3000 字的分析报告。
================================== Ai Message ==================================
Tool Calls:
  write_tasks (call_JiFxZzKPA98Se4e0UmUAlfo8)
 Call ID: call_JiFxZzKPA98Se4e0UmUAlfo8
  Args:
    tasks: [{'content': '调研 LangGraph 技术架构、核心概念与适用场景', 'depends_on': [], 'id': 'research_langgraph', 'status': 'in_progress'}, {'content': '调研并选定三个竞品的技术架构与定位', 'depends_on': [], 'id': 'research_competitors', 'status': 'in_progress'}, {'content': '综合对比并撰写约 3000 字中文分析报告', 'depends_on': ['research_langgraph', 'research_competitors'], 'id': 'write_report', 'status': 'pending'}]
================================= Tool Message =================================
Name: write_tasks

Updated task graph to [{'id': 'research_langgraph', 'content': '调研 LangGraph 技术架构、核心概念与适用场景', 'status': 'in_progress', 'depends_on': []}, {'id': 'research_competitors', 'content': '调研并选定三个竞品的技术架构与定位', 'status': 'in_progress', 'depends_on': []}, {'id': 'write_report', 'content': '综合对比并撰写约 3000 字中文分析报告', 'status': 'pending', 'depends_on': ['research_langgraph', 'research_competitors']}].
================================== Ai Message ==================================
Tool Calls:
  task (call_M96IqsC4v5nL7h8AlNzb4agt)
 Call ID: call_M96IqsC4v5nL7h8AlNzb4agt
  Args:
    description: 请调研 LangGraph 的技术架构并输出中文要点。重点包括：项目定位、核心抽象（StateGraph、节点、边、状态、检查点/持久化、人类介入、流式执行、多智能体）、运行时机制、与 LangChain/LangSmith 的关系、部署形态、优势与局限。请给出结构化摘要，便于主线程写一份约3000字中文分析报告。无需写完整报告。
    subagent_type: general-purpose
  task (call_nxkKlcrVHZnjoPjeZpRSh6uV)
 Call ID: call_nxkKlcrVHZnjoPjeZpRSh6uV
  Args:
    description: 请调研 LangGraph 的三个主要竞品/可替代方案，建议选择 AutoGen、CrewAI、LlamaIndex Workflows/Agents（如你认为更合适可替换，但需说明）。对每个竞品输出：项目定位、核心架构/抽象、编排方式、状态与持久化能力、工具/人类介入/多智能体支持、部署与生态、优劣势。最后给出与 LangGraph 的简要对比表要点。请用中文结构化输出，便于主线程撰写约3000字分析报告。
    subagent_type: general-purpose
================================= Tool Message =================================
Name: task

以下为 LangGraph 技术架构调研要点摘要，可作为撰写约 3000 字中文分析报告的素材框架。
....
================================= Tool Message =================================
Name: task

以下调研选择 **AutoGen、CrewAI、LlamaIndex Workflows/Agents** 作为 LangGraph 的三个主要竞品/可替代方案。选择理由是：三者都能覆盖“Agent 编排、多智能体协作、工具调用、人工介入、应用落地”等核心场景，但设计哲学与 LangGraph 明显不同，适合作为分析报告中的横向对比对象。

---
...
================================== Ai Message ==================================
Tool Calls:
  write_tasks (call_ApXfCSBgDZyKpZx2ATi1AYjU)
 Call ID: call_ApXfCSBgDZyKpZx2ATi1AYjU
  Args:
    tasks: [{'content': '调研 LangGraph 技术架构、核心概念与适用场景', 'depends_on': [], 'id': 'research_langgraph', 'status': 'completed'}, {'content': '调研并选定三个竞品的技术架构与定位', 'depends_on': [], 'id': 'research_competitors', 'status': 'completed'}, {'content': '综合对比并撰写约 3000 字中文分析报告', 'depends_on': ['research_langgraph', 'research_competitors'], 'id': 'write_report', 'status': 'in_progress'}]
================================= Tool Message =================================
Name: write_tasks

Updated task graph to [{'id': 'research_langgraph', 'content': '调研 LangGraph 技术架构、核心概念与适用场景', 'status': 'completed', 'depends_on': []}, {'id': 'research_competitors', 'content': '调研并选定三个竞品的技术架构与定位', 'status': 'completed', 'depends_on': []}, {'id': 'write_report', 'content': '综合对比并撰写约 3000 字中文分析报告', 'status': 'in_progress', 'depends_on': ['research_langgraph', 'research_competitors']}].
================================== Ai Message ==================================
Tool Calls:
  write_tasks (call_AXrsrzV09qUZMG8AxQ3q4HTB)
 Call ID: call_AXrsrzV09qUZMG8AxQ3q4HTB
  Args:
    tasks: [{'content': '调研 LangGraph 技术架构、核心概念与适用场景', 'depends_on': [], 'id': 'research_langgraph', 'status': 'completed'}, {'content': '调研并选定三个竞品的技术架构与定位', 'depends_on': [], 'id': 'research_competitors', 'status': 'completed'}, {'content': '综合对比并撰写约 3000 字中文分析报告', 'depends_on': ['research_langgraph', 'research_competitors'], 'id': 'write_report', 'status': 'completed'}]

================================= Tool Message =================================
Name: write_tasks

Updated task graph to [{'id': 'research_langgraph', 'content': '调研 LangGraph 技术架构、核心概念与适用场景', 'status': 'completed', 'depends_on': []}, {'id': 'research_competitors', 'content': '调研并选定三个竞品的技术架构与定位', 'status': 'completed', 'depends_on': []}, {'id': 'write_report', 'content': '综合对比并撰写约 3000 字中文分析报告', 'status': 'completed', 'depends_on': ['research_langgraph', 'research_competitors']}].
================================== Ai Message ==================================

# LangGraph 技术架构与竞品对比分析报告
...
```

# SubAgent（SubAgentMiddleware）
上面说了使用`TodoListMiddleware` 中间件进行任务分解和规划，这里我们使用`SubAgentMiddleware`来委派任务。采用subagent的形式，可以实现上下文隔离，子agent只需要把一个复杂任务的结果返回给主agent即可，这样可以让主agent的上下文保持干净。

deepagents中的`create_deep_agent` api可以会默认创建一个通用的subagent，同样我们也可以创建专门针对某个任务的subagent

```python
research_subagent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions",
    "system_prompt": "You are a great researcher",
    "tools": [internet_search],
    "model": "openai:gpt-5.4",  # Optional override, defaults to main agent model
}
```

![](https://img.leftover.cn/img-md/202606012300052.png)

## 原理
`subagent`的原理也很简单，本质上就是工具调用，`SubAgentMiddleware`中间件中会添加一个`task`工具，

调用工具的时候需要传`任务的描述` 以及`subagent的类型`，即上面定义的subagent的name

之后调用将任务描述给到subagent，然后subagent返回最终的结果，最终`task`工具把subagent的结果作为`task`工具调用的结果返回给主agent

**工具的定义：**

```python
    def task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        if subagent_type not in subagent_graphs:
            allowed_types = ", ".join([f"`{k}`" for k in subagent_graphs])
            return f"We cannot invoke subagent {subagent_type} because it does not exist, the only allowed types are {allowed_types}"
        if not runtime.tool_call_id:
            value_error_msg = "Tool call ID is required for subagent invocation"
            raise ValueError(value_error_msg)
        subagent, subagent_state = _validate_and_prepare_state(subagent_type, description, runtime)
        subagent_config = _build_subagent_config(runtime)
        # Tag the subagent's configurable so downstream readers (e.g. middleware
        # that key off `runtime.config["configurable"]["ls_agent_type"]`) see the
        # subagent context, in addition to the langsmith tracing-context tag.
        subagent_config["configurable"] = {
            **subagent_config.get("configurable", {}),
            "ls_agent_type": "subagent",
        }
        with _subagent_tracing_context():
            # 调用subagent获得结果
            result = subagent.invoke(subagent_state, subagent_config)
        # 将结果作为工具调用结果返回给主agent
        return _return_command_with_state_update(result, runtime.tool_call_id)
```

这是`SubAgentMiddleware`的`wrap_model_call`hook，他会将有关子agent调用的提示词添加到系统提示词后面，告诉他什么时候用`task`工具，什么时候不用`task`工具，以及告诉他有哪些subagent，方便主agent在工具调用的时候传入对应的subagent的类型

```python
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Update the system message to include instructions on using subagents."""
        if self.system_prompt is not None:
            new_system_message = append_to_system_message(request.system_message, self.system_prompt)
            return handler(request.override(system_message=new_system_message))
        return handler(request)
```



## 举例说明
这是上面`TodoListMiddleware`结果的一部分，主agent发起了两个`task`工具调用，这说明subagent是可以并行执行的，现在主流的模型都支持并行工具调用。最后两个任务的结果作为工具调用的结果返回回来。**因此这里主agent是阻塞的状态，他需要等待子agent返回结果**

```python
================================== Ai Message ==================================
Tool Calls:
  task (call_8w0WB2nP1AAvCBlqVvfSnbH6)
 Call ID: call_8w0WB2nP1AAvCBlqVvfSnbH6
  Args:
    description: 请调研 LangGraph 的技术架构，聚焦截至 2024/2025 的公开信息。输出中文要点，包含：核心定位、Graph/State/Node/Edge/Conditional Edge、Pregel/BSP 执行模型或 super-step 思想、checkpoint/persistence、human-in-the-loop、memory/store、streaming、subgraph、LangGraph Platform/Studio/Cloud/Server（如有）、与 LangChain 关系、优劣势和典型场景。要求结构化、准确、可用于撰写 3000 字中文分析报告。
    subagent_type: general-purpose
  task (call_cnD2toMT7hjlubDmes6Jsdnf)
 Call ID: call_cnD2toMT7hjlubDmes6Jsdnf
  Args:
    description: 请调研 LangGraph 的三个竞品/替代方案并对比，建议选择 CrewAI、Microsoft AutoGen、LlamaIndex Workflows/Agents（如认为更合适可说明）。输出中文要点，包含每个产品的核心定位、架构抽象、编排方式、状态与持久化、人机协同、可观测/部署生态、适合场景、相对 LangGraph 的优势劣势。最后给一个对比表的文字版。要求结构化、准确、可用于撰写 3000 字中文分析报告。
    subagent_type: general-purpose
================================= Tool Message =================================
Name: task

以下为截至 2024/2025 年公开信息整理的 **LangGraph 技术架构调研要点**，可直接作为撰写约 3000 字中文分析报告的素材框架。
...

================================= Tool Message =================================
Name: task

以下调研以 **LangGraph** 作为参照对象，对三个常见竞品/替代方案进行对比：**CrewAI、Microsoft AutoGen、LlamaIndex Workflows/Agents**。三者都可用于构建 Agent 应用，但设计重心不同：CrewAI 更偏“角色化多智能体协作”，AutoGen 更偏“可编程的多智能体对话与实验框架”，LlamaIndex Workflows/Agents 更偏“围绕数据/RAG 的 Agent 与工作流编排”。如果目标是构建生产级、状态可控、可恢复的人机协同 Agent 系统，LangGraph 仍然是最强参照之一。
...
```

# AsyncSubAgent(AsyncSubAgentMiddleware)
上面说到subagent是阻塞的，因为主agent需要等待`task`（子agent）的结果，但是多个子agent可以并行执行，这是因为可以并行地进行工具调用（可以同时调用多个工具）。

而AsyncSubAgent 是异步的子agent，他不是阻塞的。因为它不需要等待子agent的结果返回。

基本流程：

1、 主agent会调用`start_async_task`工具，该工具返回一个`task_id` 

2、主agent可以使用`check_async_task`工具检查任务状态，如果任务完成了，还会返回任务的结果，这里需要注意的是，主agent通常并不会主动调用这个工具，通常我们可以

+ 用户主要跟主agent说检查某个任务的状态，主agent调用`check_async_task`工具返回结果
+ 如果当前还有子agent的结果没有返回,我们可以隔几分钟/隔几轮对话，我们添加一条提示消息，提示主agent别忘了查询任务的状态

> 感觉claude code很喜欢这样做
>

3. 用户觉得子agent跑偏了,用户可以添加消息纠正其行为.codex好像就有这个功能,codex可以打断他并纠正其行为

> 具体的实现的话,打断主agent，纠正他的方向比较简单，我们可以取消当前模型的对话，并将用户的指令添加到messages的末尾即可。
>
> 如果需要打断某个子agent，可能需要界面上下点工具，比如要具体让用户选择指挥具体的那个subagent，然后我们得在末尾添加一些提示词稍微提醒一下模型调用`update_async_task`这个工具。
>

AsyncSubAgent 会添加五个tools，分别是：

| `start_async_task` | 启动一个新的后台任务 | 立刻返回task_id |
 --- | --- | --- |
| `check_async_task` | 查询任务当前状态与结果 | 返回任务状态 + 结果（若完成） |
| `update_async_task` | 根据所给的task_id，给运行中的任务追加新指令，将中断AsyncSubAgent的运行，并将新的指令添加到对话末尾，之后AsyncSubAgent重新运行 | 返回更新成功/失败的工具调用消息 |
| `cancel_async_task` | 根据所给的task_id，终止运行中的任务 | 返回取消成功/失败的工具调用消息 |
| `list_async_tasks` | 列出所有任务（含实时状态） | 返回所有的任务总览 |


基本原理和subagent差不多，只不过提供的工具有差别，所以导致了行为的差别，`wrap_model_call`的hook也是一样的，所添加的系统提示词会稍有区别

```python
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Update the system message to include async subagent instructions."""
        if self.system_prompt is not None:
            new_system_message = append_to_system_message(request.system_message, self.system_prompt)
            return handler(request.override(system_message=new_system_message))
        return handler(request)
```
