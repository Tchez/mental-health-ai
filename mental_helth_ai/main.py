from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rich import print

from mental_helth_ai.rag.database.weaviate_impl import WeaviateClient
from mental_helth_ai.rag.llm.openai_impl import OpenAILLM
from mental_helth_ai.rag.rag import RAGFactory

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

vector_db = WeaviateClient()
llm = OpenAILLM()
rag_factory = RAGFactory(vector_db=vector_db, llm=llm)


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    response: str
    source_documents: list


@app.get('/')
async def get():
    html = """<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mental Health Chatbot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f9;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        .chat-container {
            width: 100%;
            max-width: 800px;
            height: 80%;
            background: white;
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            padding: 10px;
        }

        .chat-box {
            flex: 1;
            padding: 10px;
            overflow-y: auto;
            margin-bottom: 10px;
            border-radius: 10px;
            background-color: #f9f9f9;
        }

        .chat-box .message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
            max-width: 95%;
            word-wrap: break-word;
        }

        .chat-box .message.user {
            background-color: #5c67f2;
            color: white;
            align-self: flex-end;
        }

        .chat-box .message.bot {
            background-color: #e4e6eb;
            color: #333;
            align-self: flex-start;
        }

        .input-box {
            display: flex;
            padding: 10px;
            background: #f1f1f1;
            border-top: 1px solid #ddd;
            border-radius: 10px;
            max-width: 100%;
            box-sizing: border-box;
        }

        .input-box input {
            flex: 1;
            padding: 15px;
            border: none;
            border-radius: 5px;
            margin-right: 10px;
            font-size: 16px;
            width: 100%;
        }

        .input-box button {
            padding: 15px 20px;
            background-color: #5c67f2;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }

        .loader {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #5c67f2;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }

        @keyframes spin {
            0% {
                transform: rotate(0deg);
            }

            100% {
                transform: rotate(360deg);
            }
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>

<body>
    <div class="chat-container">
        <div class="chat-box" id="chat-box"></div>
        <div class="input-box">
            <input type="text" id="user-input" placeholder="Type your message here...">
            <button id="send-button">Send</button>
        </div>
    </div>

    <script>
        const chatBox = document.getElementById('chat-box');
        const userInput = document.getElementById('user-input');
        const sendButton = document.getElementById('send-button');

        function appendMessage(message, sender) {
            const messageElement = document.createElement('div');
            messageElement.classList.add('message', sender);

            if (sender === 'bot') {
                messageElement.innerHTML = marked.parse(message);
            } else {
                messageElement.textContent = message;
            }

            chatBox.appendChild(messageElement);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        function addLoader() {
            const loader = document.createElement('div');
            loader.classList.add('loader');
            loader.id = 'loading';
            chatBox.appendChild(loader);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        function removeLoader() {
            const loader = document.getElementById('loading');
            if (loader) {
                loader.remove();
            }
        }

        async function sendMessage() {
            const message = userInput.value;
            if (!message) return;

            appendMessage(message, 'user');
            userInput.value = '';

            addLoader();

            try {
                const response = await fetch('http://127.0.0.1:8000/rag/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: message }),
                });

                const data = await response.json();
                console.log(data);
                const botMessage = data.response || 'Sorry, I didn’t understand that.';

                removeLoader();
                appendMessage(botMessage, 'bot');
            } catch (error) {
                console.error('Error:', error);
                removeLoader();
                appendMessage('Error: Could not reach the server.', 'bot');
            }
        }

        sendButton.addEventListener('click', sendMessage);

        userInput.addEventListener('keypress', function (e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>

</html>"""  # noqa: E501
    return HTMLResponse(html)


@app.post('/rag/query', response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        response, source_documents = rag_factory.generate_response(
            request.query, request.top_k
        )

        return QueryResponse(
            response=response, source_documents=source_documents
        )
    except Exception as e:
        print(f'[red]Error: {e}[/red]')
        raise HTTPException(status_code=500, detail=str(e))


# TODO: Adicionar implementação para WebSocket
