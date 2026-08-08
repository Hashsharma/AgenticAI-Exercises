from typing import TypedDict, List, Any
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
import json
import os


class AgentState(TypedDict):
    messages: List[BaseMessage]
    plan: List[dict]
    results: List[Any]


@tool
def add(a: int, b: int):
    """Add two numbers."""
    return a + b


@tool
def multiply(a: int, b: int):
    """Multiply two numbers."""
    return a * b

@tool
def subtraction(a: int, b: int):
    """Subtract two numbers."""
    return a - b



tools = {
    "add": add,
    "multiply": multiply,
    "subtract": subtraction
}


model = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL"),
    temperature=0.3,
)


def planner_node(state: AgentState):

    response = model.invoke([
            SystemMessage(
                content="""
                        You are a planner.

                        Create a JSON list of steps.

                        Available tools:
                        - add(a,b)
                        - multiply(a,b)

                        Format:

                        [
                        {
                        "task": "...",
                        "tool": "...",
                        "args": {...}
                        }
                        ]

                        Return JSON only.
                        """
                                    )
        ]
        + state["messages"]
    )
    print(response.content)
    return {
        **state,
        "plan": json.loads(response.content)
    }


def executor_node(state: AgentState):

    results = []

    for step in state["plan"]:

        tool_name = step["tool"]

        tool = tools[tool_name]

        result = tool.invoke(
            step["args"]
        )

        results.append(result)


    return {
        **state,
        "results": results
    }



def finalizer_node(state: AgentState):

    response = model.invoke(
        f"""
            User question:
            {state['messages'][0].content}

            Plan:
            {state['plan']}

            Results:
            {state['results']}


            Give the final answer.
            """)
    print(state["messages"] + [response])


    return {
        **state,
        "messages": state["messages"] + [response]
    }



graph = StateGraph(AgentState)

graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("finalizer", finalizer_node)


graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor", "finalizer")
graph.add_edge("finalizer", END)


app = graph.compile()



result = app.invoke(
    {
        "messages":[
            HumanMessage(
                content="What is 25 multiplied by 4?"
            )
        ],
        "plan":[],
        "results":[]
    }
)


print(result["messages"][-1].content)
