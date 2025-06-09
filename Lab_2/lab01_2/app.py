import os
import sys
import asyncio
import nest_asyncio
from dotenv import load_dotenv
import streamlit as st
import base64
from streamlit_chat import message as st_message
import time
from scrap.scrapper import WebScrapper
from rag.summarization import WebSummarizer
from rag.ingest import EmbeddingIngestor
from rag.chatbot import ChatBot

# Настройка для Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

load_dotenv()
nest_asyncio.apply()

MODELS = {
    "Deepseek": "deepseek-r1:1.5b",
    "Qwen": "qwen2.5:1.5b",
    "Llama": "llama3.2:3b",
    "Hermes": "hermes3:3b",
}

# Сессионные переменные
if "model" not in st.session_state:
    st.session_state.model = MODELS["Qwen"]
if "url_submitted" not in st.session_state:
    st.session_state.url_submitted = False
if "extraction_done" not in st.session_state:
    st.session_state.extraction_done = False
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "embedding_done" not in st.session_state:
    st.session_state.embedding_done = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summary" not in st.session_state:
    st.session_state.summary = ""

st.set_page_config(layout="wide", page_title="Web-ChatBot")
st.title("🤖 Chatbot с использованием LLM + RAG")

page = st.sidebar.selectbox("Menu", ["Home", "Deepseek", "Qwen", "Llama", "Hermes"])


# Функция для загрузки GIF-файла
def load_image_as_base64(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# Получаем путь к текущему скрипту
current_dir = os.path.dirname(os.path.abspath(__file__))

# Пути к GIF-файлам
bot_gif_path = os.path.join(current_dir, "images", "bot.gif")
ai_gif_path = os.path.join(current_dir, "images", "ai.gif")
robot_gif_path = os.path.join(current_dir, "images", "robot.gif")

# Загружаем GIF-файлы
try:
    bot_gif = load_image_as_base64(bot_gif_path)
    ai_gif = load_image_as_base64(ai_gif_path)
    robot_gif = load_image_as_base64(robot_gif_path)
except FileNotFoundError as e:
    st.error(f"Ошибка: {e}")

if page == "Home":
    st.markdown("""
<style>
.welcome-box {
    background: linear-gradient(135deg, #1e1e2f, #2a2a40);
    padding: 30px;
    border-radius: 15px;
    color: white;
    box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    font-family: 'Segoe UI', sans-serif;
}
.welcome-box h2 {
    color: #00ffd5;
    border-left: 5px solid #ff9900;
    padding-left: 10px;
    margin-bottom: 20px;
}
.welcome-box ul {
    list-style: none;
    padding-left: 20px;
}
.welcome-box li::before {
    content: "✨";
    margin-right: 8px;
}
.tech-title {
    color: #ff9900;
    font-weight: bold;
}
.bot-name {
    color: #00ffd5;
    font-weight: bold;
}
</style>
<div class="welcome-box">
    <h2>🤖 Добро пожаловать в <span class="bot-name">Web-ChatBot</span>!</h2>
     Это ваш личный цифровой ассистент, собранный по последнему слову техники.
    🔧 Основные функции:
    🕷️ Веб-скрапинг
    📝 Суммаризация контента
    🧠 Создание эмбеддингов
    💬 Интерфейс чат-бота
    🛠 Используемые технологии:
            LLM: deepseek-r1:1.5b, qwen2.5:1.5b, llama3.2:3b, hermes3:3b
    Приступим к работе!
    🤖 P.S. Он иногда ошибается, но всегда уверен в себе
""", unsafe_allow_html=True)

    # Отображение гифок
    st.markdown(f"""
    <div style="text-align: center; margin: 20px;">
        <img src="data:image/gif;base64,{bot_gif}" width="300" style="border-radius: 12px; margin-bottom: 20px;">
        <img src="data:image/gif;base64,{ai_gif}" width="300" style="border-radius: 12px;">
    </div>
    """, unsafe_allow_html=True)
else:
    st.session_state.model = MODELS[page]
    with st.sidebar:
        model_name = st.session_state.model
        st.sidebar.markdown(
            f"""
    <style>
    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(-10px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    .fade-in {{
        animation: fadeIn 0.6s ease forwards;
        font-size: 18px;
        color: #6a1b9a;
        font-weight: bold;
    }}
    </style>
    <div class="fade-in">Model: <code style="color:#e67e22;">{st.session_state.model}</code> 🤖</div>
    """,
            unsafe_allow_html=True
        )
        # Информация о функциях
        st.markdown("""
    <div style="margin-top: 20px; font-size: 14px; color: #ddd;">
        <strong>Что умею:</strong><br>
        🕵️‍♂️ Парсинг сайтов — ловлю всё, даже где ничего нет<br>
        📝 Суммаризация — коротко о главном<br>
        🧠 RAG + FAISS — как шифровка знаний<br>
        💬 Чат-бот — отвечаю, как будто всё знаю 😎
    </div>
""", unsafe_allow_html=True)
        # Авторы
        st.markdown("""
        <div style="margin-top: 20px; font-size: 14px; color: #fff; text-align: left;">
            <strong>💻 Наша команда:</strong><br>
            &nbsp;&nbsp;&nbsp;&nbsp; Сиволобов Александр<br>
            &nbsp;&nbsp;&nbsp;&nbsp; Голдобин Кирилл<br>
            &nbsp;&nbsp;&nbsp;&nbsp; Чумак Артемий<br>
            &nbsp;&nbsp;&nbsp;&nbsp; Севоян Ваграм
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-top: 20px; font-size: 12px; color: #aaa; text-align: center;">
            © 2025 Web-ChatBot<br>by Наша команда
        </div>
        """, unsafe_allow_html=True)
        # Отображение робота
        st.markdown(
            f'<img src="data:image/gif;base64,{robot_gif}" width="100" style="border-radius: 12px; display: block; margin: 0 auto;">',
            unsafe_allow_html=True
        )

        with open(r"C:\PC\PycharmProjects\Lab_2\lab01_2\images\robot.gif", "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(
            f'<img src="data:image/gif;base64,{data}" width="100" style="border-radius: 12px; display: block; margin: 0 auto;">',
            unsafe_allow_html=True
        )
    with st.form("url_form"):
        url_input = st.text_input("Enter a URL to crawl:")
        submit_url = st.form_submit_button("Submit URL")

        if submit_url and url_input:
            st.session_state.url_submitted = True
            st.session_state.extraction_done = False
            st.session_state.embedding_done = False
            st.session_state.chat_history = []
            st.session_state.summary = ""

    if st.session_state.url_submitted:
        col1, col2 = st.columns(2)

        with col1:
            st.header("1. Web-Scrapping")

            if not st.session_state.extraction_done:
                start_time = time.time()
                with st.spinner("Extracting website..."):
                    scraper = WebScrapper()
                    extracted = asyncio.run(scraper.crawl(url_input))
                    st.session_state.extracted_text = extracted
                    st.session_state.extraction_done = True
                elapsed = time.time() - start_time
                st.success(f"Extraction complete! Time: {elapsed:.2f}s")

            preview = "\n".join([line for line in st.session_state.extracted_text.splitlines() if line.strip()][:5])
            st.text_area("Extracted Text Preview", preview, height=150)

            st.download_button(
                label="Download Extracted Text",
                data=st.session_state.extracted_text,
                file_name="extract_text.txt",
                mime="text/plain",
            )

            st.markdown("---")

            st.header("2. Web-Summarization")

            if st.button("Summarize Web Page", key="summarize_button"):
                start_time = time.time()
                with st.spinner("Summarizing..."):
                    summarizer = WebSummarizer()
                    st.session_state.summary = summarizer.summarize(st.session_state.extracted_text)
                elapsed = time.time() - start_time
                st.success(f"Summarization complete! Time: {elapsed:.2f}s")

            if st.session_state.summary:
                st.subheader("Summarized Output")
                st.markdown(st.session_state.summary, unsafe_allow_html=False)

        with col2:
            st.header("3. Create Embeddings")

            if st.session_state.extraction_done and not st.session_state.embedding_done:
                if st.button("Create Embeddings"):
                    start_time = time.time()
                    with st.spinner("Creating embeddings..."):
                        embeddings = EmbeddingIngestor()
                        st.session_state.vectorstore = embeddings.create_embeddings(st.session_state.extracted_text)
                        st.session_state.embedding_done = True
                    elapsed = time.time() - start_time
                    st.success(f"Vectors are created! Time: {elapsed:.2f}s")

            elif st.session_state.embedding_done:
                st.info("Embeddings have been created.")

            st.markdown("---")

            st.header("4. ChatBot")

            if st.session_state.embedding_done:
                chatbot = ChatBot(st.session_state.vectorstore)
                user_input = st.text_input("Your Message:", key="chat_input")

                if st.button("Send", key="send_button") and user_input:
                    start_time = time.time()
                    bot_answer = chatbot.qa(user_input)
                    elapsed = time.time() - start_time

                    # Добавляем в историю
                    st.session_state.chat_history.append({
                        "user": user_input,
                        "bot": bot_answer,
                        "time": elapsed
                    })

                    # Сохраняем всю историю в файл
                    chat_file_content = "\n\n".join([
                        f"User: {chat['user']}\nBot: {chat['bot']}\nTime: {chat.get('time', 0):.2f}s"
                        for chat in st.session_state.chat_history
                    ])
                    with open("history/chat_history.txt", "w", encoding="utf-8") as cf:
                        cf.write(chat_file_content)
                if st.session_state.chat_history:
                    for chat in st.session_state.chat_history:
                        st_message(chat["user"], is_user=True)
                        st_message(
                            f"{chat['bot']}\n\n {chat.get('time', 0):.2f}s",
                            is_user=False
                        )

            else:
                st.info("Please create embeddings to activate the chat.")
