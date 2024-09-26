import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Cargar las variables de entorno
load_dotenv()

# Obtener la clave API de OpenAI desde el archivo .env
openai_api_key = os.getenv("OPENAI_API_KEY")

# Configuración de la app
st.set_page_config(page_title="ChatBot", page_icon="🤖")
st.title("Casita IA")

# Función para obtener la respuesta del modelo
def get_response(user_query, chat_history):
    template = """
    Eres una inteligencia artificial de nombre Casita, responde a las preguntas considerando la historia de la conversacion:
    Chat history: {chat_history}

    User question: {user_question}
    """

    # Generar el prompt
    prompt = ChatPromptTemplate.from_template(template)

    # Crear la instancia de ChatOpenAI con la clave API
    llm = ChatOpenAI(api_key=openai_api_key)

    # Crear el chain con prompt, llm y output parser
    chain = prompt | llm | StrOutputParser()

    # Prueba usando invoke o predict
    try:
        result = chain.invoke({
            "chat_history": chat_history,
            "user_question": user_query,
        })
    except AttributeError:
        result = chain.predict({
            "chat_history": chat_history,
            "user_question": user_query,
        })

    return result

# Estado de sesión para el historial del chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hola soy Casita, como puedo ayudarte?"),
    ]

# Mostrar el historial de la conversación
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# Captura de la entrada del usuario
user_query = st.chat_input("Escribe tu mensaje...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    # Obtener la respuesta de la IA
    response = get_response(user_query, st.session_state.chat_history)

    # Mostrar la respuesta de la IA
    with st.chat_message("AI"):
        st.markdown(response)

    st.session_state.chat_history.append(AIMessage(content=response))
