import streamlit as st
from tts import speak
import speech_recognition as sr
with st.spinner("Loading College Chatbot..."):
    from RAG import get_graph
    graph = get_graph()


st.title("College Chatbot")
st.set_page_config(
    page_title="College Chatbot",
    page_icon="🎓",
    layout="wide"
)

# ------------------------
# Chat History
# ------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages

for message in st.session_state.messages:

    if message.type == "human":
        with st.chat_message("user"):
            st.write(message.content)

    elif message.type == "ai":

        # Skip AI messages that are only tool calls
        if getattr(message, "tool_calls", None):
            continue
        
        if not message.text.strip():
            continue
        
        with st.chat_message("assistant"):
            st.write(message.text)
# ------------------------
# User Input
# ------------------------

typed_query = st.chat_input("Ask me anything...")
voice_query = None
if st.button("🎤 Speak"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        voice_query = r.recognize_google(audio)
    except Exception:
        st.error("Couldn't recognize speech.")
query = typed_query if typed_query else voice_query
if query:
    # Display user message
    with st.chat_message("user"):
        st.write(query)

    # Add user message to history
    st.session_state.messages.append(
        ("user", query)
    )

    # Run graph
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            response = graph.invoke(
                {
                    "messages": st.session_state.messages
                }
            )

            ai_message = response["messages"][-1]

            st.write(ai_message.text)
            speak(ai_message.text.split("\n")[0])

    # Save updated history
    st.session_state.messages = response["messages"]