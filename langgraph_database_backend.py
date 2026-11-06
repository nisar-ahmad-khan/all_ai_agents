import os
import sqlite3 #imported this library to use sqlite database
from langgraph.checkpoint.sqlite import SqliteSaver #installed this llibrary using pip install langgraph-checkpoint-sqlite
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
# from langgraph.checkpoint.memory import MemorySaver  #it is used to save conversations in RAM
# from langgraph.checkpoint.memory import InMemorySaver 
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

conn = sqlite3.connect(database='agent_db.db',check_same_thread=False) #a connection to the SQLite database is established using the sqlite3 library. The database file is named 'agent_db.db', and the check_same_thread parameter is set to False, allowing multiple threads to access the database connection simultaneously.
checkpointer = SqliteSaver(conn = conn) #used to save the conversation state in a SQLite database. It takes the database connection conn as an argument, allowing it to interact with the SQLite database for storing and retrieving conversation data.
graph = StateGraph(ChatState)

graph.add_node("chat", chat_node)


graph.add_edge(START,"chat")
graph.add_edge("chat",END)

workflow = graph.compile(checkpointer=checkpointer)


#test:
# config = {'configurable': {'thread_id': 'thread_1'}}
# response = workflow.invoke(
#                 {'messages': [HumanMessage(content="what do you know about me?")]},
#                 config=config,
#             )
# print(response)
def retrieve_all_threads():#added this function to retrieve all the thread_ids from the database
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)