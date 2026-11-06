import os
import sqlite3
from typing import TypedDict, Annotated, Optional
import requests

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(find_dotenv(), override=True)

# Step 1: Model Instance
model = ChatOpenAI(
    model=os.getenv("KIMI_MODEL"),
    base_url=os.getenv("KIMI_BASE_URL").rstrip('/') + "/v1",
    api_key="no-need",
    default_headers={
        "Modal-key": os.getenv("MODEL_KEY"),
        "Modal-Secret": os.getenv("MODEL_SECRET")
    }
)

# Step 2: Tools
ddg_raw = DuckDuckGoSearchRun(region="us-en")

@tool
def search_tool(query: str) -> str:
    """Searches the web for non-Eventlay queries or general information."""
    try:
        return ddg_raw.run(query)
    except Exception as e:
        return f"Web search unavailable: {str(e)}. Proceeding with base context."

@tool
def calculator(first_num: float, second_num: float, operator: str) -> dict:
    """Performs basic arithmetic operations (+, -, *, /)."""
    if operator == '+':
        return {"result": first_num + second_num}
    elif operator == '-':
        return {"result": first_num - second_num}
    elif operator == '*':
        return {"result": first_num * second_num}
    elif operator == '/':
        if second_num == 0:
            return {"error": "Division by zero is not allowed."}
        return {"result": first_num / second_num}
    return {"error": "Invalid operator."}

@tool
def get_event_info(
    query: Optional[str] = None, 
    city: Optional[str] = None, 
    category: Optional[str] = None,
    limit: int = 5
) -> dict:
    """
    Fetches real-time event data from Eventlay (eventlay.com).
    Use this tool whenever users ask about upcoming events, workshops, tech meetups, 
    concerts, schedules, venues, or ticket details on Eventlay.

    Args:
        query: Search term (e.g., 'workshop', 'python', 'music', or specific event name).
        city: City name filter (e.g., 'Lahore', 'Karachi', 'Islamabad').
        category: Category name filter (e.g., 'Technology', 'Business', 'Music').
        limit: Max number of event records to return (default is 5).
    """
    url = "https://admin.eventlay.com/api/get-events"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Standardize payload extraction
        raw_events = data.get("events", data.get("data", data)) if isinstance(data, dict) else data
        if not isinstance(raw_events, list):
            return {"error": "Invalid API payload structure from Eventlay."}

        filtered = []
        q_clean = query.lower().strip() if query else ""
        city_clean = city.lower().strip() if city else ""
        cat_clean = category.lower().strip() if category else ""

        for evt in raw_events:
            if not isinstance(evt, dict):
                continue
            
            title = str(evt.get("title", "")).lower()
            desc = str(evt.get("description", "")).lower()
            venue = str(evt.get("venue_name", evt.get("venue", ""))).lower()
            evt_city = str(evt.get("city", evt.get("location", ""))).lower()
            evt_cat = str(evt.get("category", evt.get("type", ""))).lower()

            # Multi-field matching condition
            match_query = (not q_clean) or any(q_clean in field for field in [title, desc, venue, evt_cat])
            match_city = (not city_clean) or (city_clean in evt_city or city_clean in venue)
            match_cat = (not cat_clean) or (cat_clean in evt_cat)

            if match_query and match_city and match_cat:
                # Truncate long descriptions to fit LLM context efficiently
                clean_desc = evt.get("description", "")
                if clean_desc and len(str(clean_desc)) > 160:
                    clean_desc = str(clean_desc)[:160] + "..."

                slug_or_id = evt.get("slug", evt.get("id"))
                filtered.append({
                    "id": evt.get("id"),
                    "title": evt.get("title"),
                    "date": evt.get("date", evt.get("start_date")),
                    "time": evt.get("time", evt.get("start_time")),
                    "venue": evt.get("venue_name", evt.get("venue")),
                    "city": evt.get("city", evt.get("location")),
                    "category": evt.get("category"),
                    "price": evt.get("price", evt.get("ticket_price", "N/A")),
                    "summary": clean_desc,
                    "url": f"https://eventlay.com/events/{slug_or_id}" if slug_or_id else None
                })

                if len(filtered) >= limit:
                    break

        # Fallback if specific search returns zero records
        if not filtered:
            recent_summary = [
                {"id": e.get("id"), "title": e.get("title"), "date": e.get("date")} 
                for e in raw_events[:3] if isinstance(e, dict)
            ]
            return {
                "message": f"No exact events found matching query='{query or ''}' city='{city or ''}'.",
                "upcoming_available_events": recent_summary
            }

        return {"total_matches": len(filtered), "events": filtered}

    except Exception as e:
        return {"error": f"Failed to fetch data from Eventlay backend: {str(e)}"}

tools = [search_tool, calculator, get_event_info]
llm_with_tools = model.bind_tools(tools)

# Step 3: State & Graph Nodes
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

SYSTEM_PROMPT = SystemMessage(content=(
    "You are the official Eventlay AI Assistant (eventlay.com). "
    "Whenever a user asks about events, conferences, workshops, venues, schedules, or ticket prices, "
    "always invoke the `get_event_info` tool to query the latest backend database before answering."
))

def chat_node(state: ChatState):
    messages = state['messages']
    # Prepend system context if not already present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SYSTEM_PROMPT] + list(messages)
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools, handle_tool_errors=True)

# Step 4: Checkpointer & Workflow Setup
conn = sqlite3.connect(database='agent_db.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")

workflow = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads() -> list[str]:
    """Efficiently retrieves thread IDs directly via SQL query."""
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT thread_id FROM checkpoints;")
    return [row[0] for row in cursor.fetchall()]