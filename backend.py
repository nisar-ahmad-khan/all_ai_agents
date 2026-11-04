import os
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
# from langgraph.checkpoint.memory import MemorySaver  #it is used to save conversations in RAM
from langgraph.checkpoint.memory import InMemorySaver 
from langchain_core.messages import BaseMessage ,HumanMessage
from dotenv import load_dotenv, find_dotenv
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
load_dotenv(find_dotenv(), override=True)

model = ChatOpenAI(
    model=os.getenv("KIMI_MODEL"),
    base_url=os.getenv("KIMI_BASE_URL").rstrip('/') + "/v1",
    api_key="no-need",
    default_headers={
        "Modal-key": os.getenv("MODEL_KEY"),
        "Modal-Secret": os.getenv("MODEL_SECRET")
    }
)

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]


def chat_node(state: ChatState):
    message = state['messages']

    response = model.invoke(message)

    return {"messages" : [response]}


checkpointer = InMemorySaver()
graph = StateGraph(ChatState)

graph.add_node("chat", chat_node)


graph.add_edge(START,"chat")
graph.add_edge("chat",END)

workflow = graph.compile(checkpointer=checkpointer)
