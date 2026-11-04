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
    response = workflow.invoke({'messages':[HumanMessage(content=chat_input)]},config=config)
    ai_message = response['messages'][-1].content
    st.session_state['message_history'].append({'role':'assistant','content':ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)