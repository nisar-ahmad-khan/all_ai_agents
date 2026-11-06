import streamlit as st
from backend import workflow
from langchain_core.messages import HumanMessage


if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

    #loading the conversation history from the session state
for message in st.session_state['message_history']:
        with st.chat_message(message['role']):
            st.text(message['content'])


chat_input = st.chat_input('ask me anything...')


if chat_input:
    st.session_state['message_history'].append({'role':'user','content':chat_input})
    with st.chat_message('user'):
        st.text(chat_input)
        config = {'configurable':{'thread_id': "1"}}
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
             message_chunk.content for message_chunk , metadata in workflow.stream(
                  {'messages':[HumanMessage(content=chat_input)]},
                  config=config,stream_mode='messages')     
        )
    st.session_state['message_history'].append({'role':'assistant','content':ai_message})
        