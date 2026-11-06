import uuid
import streamlit as st
from backend import workflow
from langchain_core.messages import HumanMessage

########################### utility functions ##########################
def generate_thread_id():
    return str(uuid.uuid4())

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []

def load_conversation(thread_id):
    state = workflow.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

#########################################################################


############################ session state setup ############################
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

add_thread(st.session_state['thread_id'])
############################################################################################


st.sidebar.title("XB Chatbot")
if st.sidebar.button("New Conversation"):
    reset_chat()

st.sidebar.header("Conversation History")
for thread_id in st.session_state['chat_threads'][::-1]:  # Display threads in reverse order (most recent first)
    if st.sidebar.button(str(thread_id), key=f"btn_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})
        st.session_state['message_history'] = temp_messages


# Display conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


chat_input = st.chat_input('Ask me anything...')

if chat_input:
    st.session_state['message_history'].append({'role': 'user', 'content': chat_input})
    with st.chat_message('user'):
        st.markdown(chat_input)

    config = {'configurable': {'thread_id': st.session_state['thread_id']}}

    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content 
            for message_chunk, metadata in workflow.stream(
                {'messages': [HumanMessage(content=chat_input)]},
                config=config,
                stream_mode='messages'
            )
            if hasattr(message_chunk, 'content')
        )
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})