from flask import Flask, request, jsonify, render_template_string
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load model
print("🔄 Loading AI model... (this takes 2-3 minutes)")
MODEL_NAME = "microsoft/DialoGPT-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print("✅ AI Model loaded successfully!")

def generate_response(user_input, chat_history_ids=None):
    # Encode user input
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    
    # Append to chat history if exists
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids
    
    # Generate response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
        temperature=0.8
    )
    
    # Decode response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    return response, chat_history_ids

# Store chat histories (simple in-memory storage)
chat_histories = {}

# HTML template
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="theme-color" content="#1a1a2e">
    <title>My AI Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 10px;
        }
        
        .container {
            width: 100%;
            max-width: 800px;
            height: 90vh;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .header {
            background: rgba(0, 0, 0, 0.3);
            padding: 20px;
            text-align: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .header h1 {
            color: #00d4ff;
            font-size: 24px;
            margin-bottom: 5px;
        }
        
        .header p {
            color: #aaa;
            font-size: 14px;
        }
        
        .chat-area {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .message {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 15px;
            animation: fadeIn 0.3s ease;
            word-wrap: break-word;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .user-message {
            align-self: flex-end;
            background: linear-gradient(135deg, #00d4ff, #0099cc);
            color: white;
            border-bottom-right-radius: 5px;
        }
        
        .bot-message {
            align-self: flex-start;
            background: rgba(255, 255, 255, 0.1);
            color: #e0e0e0;
            border-bottom-left-radius: 5px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .input-area {
            padding: 20px;
            background: rgba(0, 0, 0, 0.3);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            gap: 10px;
        }
        
        .input-area input {
            flex: 1;
            padding: 12px 20px;
            border: none;
            border-radius: 25px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            font-size: 16px;
            outline: none;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .input-area input::placeholder {
            color: #888;
        }
        
        .input-area button {
            padding: 12px 25px;
            border: none;
            border-radius: 25px;
            background: linear-gradient(135deg, #00d4ff, #0099cc);
            color: white;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .input-area button:hover {
            transform: scale(1.05);
        }
        
        .input-area button:active {
            transform: scale(0.95);
        }
        
        .typing-indicator {
            align-self: flex-start;
            background: rgba(255, 255, 255, 0.1);
            padding: 15px 20px;
            border-radius: 15px;
            display: none;
        }
        
        .typing-indicator span {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #00d4ff;
            margin: 0 2px;
            animation: typing 1.4s infinite;
        }
        
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }
        
        /* Scrollbar styling */
        .chat-area::-webkit-scrollbar {
            width: 6px;
        }
        
        .chat-area::-webkit-scrollbar-track {
            background: transparent;
        }
        
        .chat-area::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 3px;
        }
        
        @media (max-width: 600px) {
            .container {
                height: 100vh;
                border-radius: 0;
            }
            .message {
                max-width: 90%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 My AI Assistant</h1>
            <p>Always online · Free · Private</p>
        </div>
        
        <div class="chat-area" id="chatArea">
            <div class="message bot-message">
                Hello! 👋 I'm your AI assistant. How can I help you today?
            </div>
        </div>
        
        <div class="typing-indicator" id="typingIndicator">
            <span></span><span></span><span></span>
        </div>
        
        <div class="input-area">
            <input type="text" id="messageInput" placeholder="Type your message here..." onkeypress="handleKeyPress(event)">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>
    
    <script>
        let sessionId = 'user_' + Math.random().toString(36).substr(2, 9);
        
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message to chat
            addMessage(message, 'user');
            input.value = '';
            
            // Show typing indicator
            document.getElementById('typingIndicator').style.display = 'block';
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        message: message,
                        session_id: sessionId
                    })
                });
                
                const data = await response.json();
                
                // Hide typing indicator
                document.getElementById('typingIndicator').style.display = 'none';
                
                // Add bot response
                addMessage(data.reply, 'bot');
                
            } catch (error) {
                document.getElementById('typingIndicator').style.display = 'none';
                addMessage('Sorry, something went wrong. Please try again.', 'bot');
            }
        }
        
        function addMessage(text, sender) {
            const chatArea = document.getElementById('chatArea');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + (sender === 'user' ? 'user-message' : 'bot-message');
            messageDiv.textContent = text;
            chatArea.appendChild(messageDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_PAGE)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        
        # Get chat history for this session
        if session_id not in chat_histories:
            chat_histories[session_id] = None
        
        # Generate response
        response, chat_histories[session_id] = generate_response(
            user_message,
            chat_histories[session_id]
        )
        
        # Clean response
        if not response or len(response.strip()) < 2:
            response = "I'm not sure what to say. Can you tell me more?"
        
        return jsonify({
            'reply': response,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'reply': 'Sorry, I encountered an error. Please try again.',
            'status': 'error'
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)