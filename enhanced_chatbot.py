"""
JnanaVerse – enhanced_chatbot.py
Advanced web chatbot with comprehensive NLP features:
- Text generation, translation, classification, similarity
- Summarization, question answering, sentiment analysis, NER
- Streaming responses, conversation history, export chat
- Dark/light mode, copy buttons, model switching
- Sanskrit transliteration support

Developer(s): Dharmin Joshi / DevKay
"""

import argparse
import threading
import webbrowser
import json
import datetime
import uuid
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, Response, stream_with_context

from jnanaverse import JnanaVerse, CustomTransformerLM, set_seed, get_device, count_parameters

# ──────────────────────────────────────────────────────────────
# Flask app + global state
# ──────────────────────────────────────────────────────────────
app = Flask(__name__)
T5_MODEL = None
CUSTOM_MODEL = None  
TRANSLATOR = None
NER_PIPELINE = None
SENTIMENT_PIPELINE = None
DEVICE = None

# Conversation sessions storage (in production, use Redis/DB)
SESSIONS = {}

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>JnanaVerse Advanced Chatbot</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
  
  :root {
    /* Dark theme (default) */
    --bg: #0a0a0a;
    --surface: #111111;
    --surface-2: #1a1a1a;
    --border: #333;
    --border-light: #444;
    --text: #e4e4e7;
    --text-dim: #a1a1aa;
    --text-muted: #71717a;
    --accent: #6366f1;
    --accent-hover: #5048e5;
    --success: #10b981;
    --warning: #f59e0b;
    --error: #ef4444;
    --user-bg: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    --bot-bg: var(--surface-2);
    --code-bg: #1e1e1e;
    --shadow: 0 4px 12px rgba(0,0,0,0.3);
    --radius: 12px;
    --radius-sm: 6px;
    --font-body: 'Space Grotesk', system-ui, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
  }
  
  [data-theme="light"] {
    --bg: #fafafa;
    --surface: #ffffff;
    --surface-2: #f5f5f5;
    --border: #e5e5e5;
    --border-light: #d4d4d8;
    --text: #18181b;
    --text-dim: #52525b;
    --text-muted: #71717a;
    --accent: #6366f1;
    --accent-hover: #5048e5;
    --user-bg: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    --bot-bg: var(--surface-2);
    --code-bg: #f8f8f8;
    --shadow: 0 4px 12px rgba(0,0,0,0.1);
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-body);
    line-height: 1.6;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
    transition: background-color 0.3s ease, color 0.3s ease;
  }

  /* Header */
  .header {
    padding: 16px 24px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: var(--shadow);
    z-index: 100;
  }
  
  .logo-section {
    display: flex;
    align-items: center;
    gap: 16px;
  }
  
  .logo {
    width: 48px;
    height: 48px;
    border-radius: var(--radius);
    background: var(--accent);
    display: grid;
    place-items: center;
    font-size: 20px;
    font-weight: 700;
    color: white;
    position: relative;
    overflow: hidden;
  }
  
  .logo::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
    transform: rotate(45deg);
    animation: shine 3s infinite;
  }
  
  @keyframes shine {
    0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
    50% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    100% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
  }
  
  .title-section h1 {
    font-size: 24px;
    font-weight: 600;
    background: linear-gradient(135deg, var(--accent), #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  
  .title-section .subtitle {
    font-size: 14px;
    color: var(--text-dim);
    font-weight: 300;
  }
  
  .header-controls {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  
  .model-selector {
    padding: 8px 16px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--surface);
    color: var(--text);
    font-family: var(--font-body);
    font-size: 14px;
    cursor: pointer;
  }
  
  .theme-toggle {
    width: 40px;
    height: 40px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--surface);
    color: var(--text);
    cursor: pointer;
    display: grid;
    place-items: center;
    font-size: 16px;
    transition: all 0.2s ease;
  }
  
  .theme-toggle:hover {
    background: var(--surface-2);
    border-color: var(--border-light);
  }

  /* Task bar */
  .task-bar {
    padding: 16px 24px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    overflow-x: auto;
  }
  
  .task-btn {
    padding: 8px 16px;
    border-radius: 20px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text-dim);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    white-space: nowrap;
    position: relative;
  }
  
  .task-btn:hover {
    border-color: var(--accent);
    color: var(--accent);
  }
  
  .task-btn.active {
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }

  /* Main chat area */
  .chat-container {
    flex: 1;
    display: flex;
    overflow: hidden;
  }
  
  .messages-area {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  
  .message {
    max-width: 80%;
    padding: 16px 20px;
    border-radius: var(--radius);
    font-size: 15px;
    line-height: 1.6;
    animation: slideIn 0.3s ease;
    position: relative;
  }
  
  @keyframes slideIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
  }
  
  .message.user {
    background: var(--user-bg);
    color: white;
    align-self: flex-end;
    border-bottom-right-radius: 4px;
  }
  
  .message.bot {
    background: var(--bot-bg);
    color: var(--text);
    align-self: flex-start;
    border-bottom-left-radius: 4px;
    border: 1px solid var(--border);
  }
  
  .message.system {
    background: transparent;
    color: var(--text-muted);
    font-size: 13px;
    text-align: center;
    align-self: center;
    max-width: 100%;
    padding: 8px 16px;
  }
  
  .message.bot.streaming {
    position: relative;
  }
  
  .message.bot.streaming::after {
    content: '▋';
    animation: blink 1s infinite;
    color: var(--accent);
  }
  
  @keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
  }
  
  .message-actions {
    display: flex;
    gap: 8px;
    margin-top: 12px;
    opacity: 0;
    transition: opacity 0.2s ease;
  }
  
  .message.bot:hover .message-actions {
    opacity: 1;
  }
  
  .action-btn {
    padding: 4px 8px;
    border: 1px solid var(--border-light);
    border-radius: var(--radius-sm);
    background: var(--surface);
    color: var(--text-dim);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  
  .action-btn:hover {
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }

  /* Code blocks */
  .code-block {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 16px;
    font-family: var(--font-mono);
    font-size: 14px;
    overflow-x: auto;
    margin: 12px 0;
  }

  /* Sidebar */
  .sidebar {
    width: 320px;
    background: var(--surface);
    border-left: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  
  .sidebar-section {
    padding: 20px;
    border-bottom: 1px solid var(--border);
  }
  
  .sidebar-title {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 16px;
    color: var(--text);
  }
  
  .history-item {
    padding: 12px;
    border-radius: var(--radius-sm);
    background: var(--surface-2);
    margin-bottom: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
    font-size: 14px;
  }
  
  .history-item:hover {
    background: var(--border);
  }
  
  .export-btn {
    width: 100%;
    padding: 12px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--surface-2);
    color: var(--text);
    font-family: var(--font-body);
    font-size: 14px;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  
  .export-btn:hover {
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }

  /* Input area */
  .input-area {
    padding: 20px 24px;
    background: var(--surface);
    border-top: 1px solid var(--border);
    display: flex;
    gap: 12px;
    align-items: flex-end;
  }
  
  .input-container {
    flex: 1;
    position: relative;
  }
  
  .input-field {
    width: 100%;
    min-height: 48px;
    max-height: 120px;
    padding: 12px 16px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-body);
    font-size: 15px;
    line-height: 1.5;
    resize: none;
    outline: none;
    transition: all 0.2s ease;
  }
  
  .input-field:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
  }
  
  .input-field::placeholder {
    color: var(--text-muted);
  }
  
  .send-btn {
    width: 48px;
    height: 48px;
    border: none;
    border-radius: var(--radius);
    background: var(--accent);
    color: white;
    font-size: 16px;
    cursor: pointer;
    transition: all 0.2s ease;
    display: grid;
    place-items: center;
  }
  
  .send-btn:hover:not(:disabled) {
    background: var(--accent-hover);
    transform: translateY(-1px);
  }
  
  .send-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
  }

  /* Responsive */
  @media (max-width: 768px) {
    .sidebar {
      position: fixed;
      top: 0;
      right: -320px;
      height: 100vh;
      z-index: 1000;
      transition: right 0.3s ease;
    }
    
    .sidebar.open {
      right: 0;
    }
    
    .message {
      max-width: 90%;
    }
    
    .header-controls {
      display: none;
    }
  }

  /* Utility classes */
  .hidden { display: none !important; }
  .loading { opacity: 0.6; pointer-events: none; }
</style>
</head>
<body>
  <div class="header">
    <div class="logo-section">
      <div class="logo">J</div>
      <div class="title-section">
        <h1>JnanaVerse</h1>
        <div class="subtitle">Advanced Multi-Task NLP • Dharmin Joshi / DevKay</div>
      </div>
    </div>
    <div class="header-controls">
      <select class="model-selector" id="modelSelect">
        <option value="t5">T5 Model</option>
        <option value="custom">Custom Transformer</option>
      </select>
      <button class="theme-toggle" id="themeToggle">🌙</button>
    </div>
  </div>

  <div class="task-bar">
    <button class="task-btn active" data-task="generation">💬 Generate</button>
    <button class="task-btn" data-task="translate">🌐 Sanskrit</button>
    <button class="task-btn" data-task="summarize">📝 Summarize</button>
    <button class="task-btn" data-task="qa">❓ Q&A</button>
    <button class="task-btn" data-task="sentiment">😊 Sentiment</button>
    <button class="task-btn" data-task="ner">🏷️ Named Entities</button>
    <button class="task-btn" data-task="classify">📊 Classify</button>
    <button class="task-btn" data-task="similarity">🔗 Similarity</button>
  </div>

  <div class="chat-container">
    <div class="messages-area" id="messages">
      <div class="message system">
        Welcome to JnanaVerse! Select a task above and start your conversation.
        <br>✨ Features: Streaming responses, conversation history, export chat, dark/light mode
      </div>
    </div>
    
    <div class="sidebar" id="sidebar">
      <div class="sidebar-section">
        <div class="sidebar-title">💾 Conversation History</div>
        <div id="historyList">
          <div class="history-item">No conversations yet</div>
        </div>
      </div>
      
      <div class="sidebar-section">
        <div class="sidebar-title">⚙️ Settings</div>
        <label style="display: block; margin-bottom: 12px; font-size: 14px;">
          Temperature: <span id="tempValue">0.8</span>
          <input type="range" id="temperature" min="0.1" max="2.0" step="0.1" value="0.8" style="width: 100%; margin-top: 4px;">
        </label>
        <label style="display: block; margin-bottom: 16px; font-size: 14px;">
          Max Tokens: <span id="tokensValue">128</span>
          <input type="range" id="maxTokens" min="32" max="512" step="16" value="128" style="width: 100%; margin-top: 4px;">
        </label>
        <button class="export-btn" id="exportBtn">📄 Export Chat</button>
      </div>
    </div>
  </div>

  <div class="input-area">
    <div class="input-container">
      <textarea 
        class="input-field" 
        id="messageInput" 
        placeholder="Type your message..." 
        rows="1"
      ></textarea>
    </div>
    <button class="send-btn" id="sendBtn">
      <span id="sendIcon">➤</span>
    </button>
  </div>

<script>
class JnanaChatbot {
  constructor() {
    this.sessionId = this.generateSessionId();
    this.currentTask = 'generation';
    this.currentModel = 't5';
    this.conversation = [];
    this.simStep = 0;
    this.simSent1 = '';
    
    this.initElements();
    this.bindEvents();
    this.loadSettings();
  }
  
  generateSessionId() {
    return 'sess_' + Math.random().toString(36).substr(2, 9);
  }
  
  initElements() {
    this.elements = {
      messages: document.getElementById('messages'),
      input: document.getElementById('messageInput'),
      sendBtn: document.getElementById('sendBtn'),
      sendIcon: document.getElementById('sendIcon'),
      themeToggle: document.getElementById('themeToggle'),
      modelSelect: document.getElementById('modelSelect'),
      taskBtns: document.querySelectorAll('.task-btn'),
      temperature: document.getElementById('temperature'),
      tempValue: document.getElementById('tempValue'),
      maxTokens: document.getElementById('maxTokens'),
      tokensValue: document.getElementById('tokensValue'),
      exportBtn: document.getElementById('exportBtn'),
      historyList: document.getElementById('historyList'),
      sidebar: document.getElementById('sidebar')
    };
  }
  
  bindEvents() {
    // Send message
    this.elements.sendBtn.addEventListener('click', () => this.sendMessage());
    this.elements.input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });
    
    // Auto-resize textarea
    this.elements.input.addEventListener('input', () => {
      this.elements.input.style.height = 'auto';
      this.elements.input.style.height = Math.min(this.elements.input.scrollHeight, 120) + 'px';
    });
    
    // Task switching
    this.elements.taskBtns.forEach(btn => {
      btn.addEventListener('click', () => this.switchTask(btn.dataset.task, btn.textContent.trim()));
    });
    
    // Theme toggle
    this.elements.themeToggle.addEventListener('click', () => this.toggleTheme());
    
    // Model selection
    this.elements.modelSelect.addEventListener('change', (e) => {
      this.currentModel = e.target.value;
      this.addSystemMessage(`Switched to: ${e.target.value === 't5' ? 'T5 Model' : 'Custom Transformer'}`);
    });
    
    // Settings
    this.elements.temperature.addEventListener('input', (e) => {
      this.elements.tempValue.textContent = e.target.value;
    });
    
    this.elements.maxTokens.addEventListener('input', (e) => {
      this.elements.tokensValue.textContent = e.target.value;
    });
    
    // Export
    this.elements.exportBtn.addEventListener('click', () => this.exportConversation());
  }
  
  loadSettings() {
    const theme = localStorage.getItem('jnana-theme') || 'dark';
    this.setTheme(theme);
  }
  
  setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    this.elements.themeToggle.textContent = theme === 'dark' ? '☀️' : '🌙';
    localStorage.setItem('jnana-theme', theme);
  }
  
  toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    this.setTheme(current === 'dark' ? 'light' : 'dark');
  }
  
  switchTask(task, label) {
    this.elements.taskBtns.forEach(btn => btn.classList.remove('active'));
    document.querySelector(`[data-task="${task}"]`).classList.add('active');
    this.currentTask = task;
    this.simStep = 0;
    this.addSystemMessage(`Switched to: ${label}`);
    
    // Update placeholder
    const placeholders = {
      generation: 'Type your message...',
      translate: 'Enter English text to translate to Sanskrit...',
      summarize: 'Paste text to summarize...',
      qa: 'Ask a question with context: Q: ... Context: ...',
      sentiment: 'Enter text to analyze sentiment...',
      ner: 'Enter text to extract named entities...',
      classify: 'Enter text to classify...',
      similarity: 'Enter first sentence for comparison...'
    };
    this.elements.input.placeholder = placeholders[task] || placeholders.generation;
    
    if (task === 'similarity') {
      this.addSystemMessage('Enter sentence 1:');
    }
  }
  
  async sendMessage() {
    const text = this.elements.input.value.trim();
    if (!text) return;
    
    this.elements.input.value = '';
    this.elements.input.style.height = 'auto';
    
    // Handle similarity task workflow
    if (this.currentTask === 'similarity') {
      if (this.simStep === 0) {
        this.addUserMessage(text);
        this.simSent1 = text;
        this.simStep = 1;
        this.addSystemMessage('Enter sentence 2:');
        return;
      } else {
        this.addUserMessage(text);
        await this.sendToAPI({
          task: 'similarity',
          s1: this.simSent1,
          s2: text,
          model: this.currentModel,
          session_id: this.sessionId
        });
        this.simStep = 0;
        this.addSystemMessage('Enter sentence 1:');
        return;
      }
    }
    
    this.addUserMessage(text);
    
    await this.sendToAPI({
      task: this.currentTask,
      message: text,
      model: this.currentModel,
      session_id: this.sessionId,
      temperature: parseFloat(this.elements.temperature.value),
      max_tokens: parseInt(this.elements.maxTokens.value)
    });
  }
  
  async sendToAPI(data) {
    this.setLoading(true);
    
    try {
      const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      
      if (response.headers.get('content-type')?.includes('text/stream')) {
        await this.handleStreamingResponse(response);
      } else {
        const result = await response.json();
        this.addBotMessage(result.response || result.error || '(No response)');
      }
    } catch (error) {
      this.addBotMessage(`Error: ${error.message}`);
    } finally {
      this.setLoading(false);
    }
  }
  
  async handleStreamingResponse(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let messageEl = this.addBotMessage('', true); // streaming = true
    let content = '';
    
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') return;
            
            try {
              const parsed = JSON.parse(data);
              if (parsed.token) {
                content += parsed.token;
                messageEl.innerHTML = this.formatMessage(content);
                this.scrollToBottom();
              }
            } catch (e) {
              // Ignore JSON parse errors for partial chunks
            }
          }
        }
      }
    } finally {
      messageEl.classList.remove('streaming');
      this.addMessageActions(messageEl, content);
    }
  }
  
  setLoading(loading) {
    this.elements.sendBtn.disabled = loading;
    this.elements.sendIcon.textContent = loading ? '⏳' : '➤';
  }
  
  addUserMessage(content) {
    const msgEl = this.createMessageElement('user', content);
    this.elements.messages.appendChild(msgEl);
    this.conversation.push({ role: 'user', content, timestamp: new Date() });
    this.scrollToBottom();
  }
  
  addBotMessage(content, streaming = false) {
    const msgEl = this.createMessageElement('bot', content);
    if (streaming) msgEl.classList.add('streaming');
    this.elements.messages.appendChild(msgEl);
    
    if (!streaming) {
      this.conversation.push({ role: 'assistant', content, timestamp: new Date() });
      this.addMessageActions(msgEl, content);
    }
    
    this.scrollToBottom();
    return msgEl;
  }
  
  addSystemMessage(content) {
    const msgEl = this.createMessageElement('system', content);
    this.elements.messages.appendChild(msgEl);
    this.scrollToBottom();
  }
  
  createMessageElement(role, content) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.innerHTML = this.formatMessage(content);
    return div;
  }
  
  formatMessage(content) {
    // Basic formatting: code blocks, line breaks
    return content
      .replace(/```([^`]+)```/g, '<div class="code-block">$1</div>')
      .replace(/`([^`]+)`/g, '<code style="background: var(--code-bg); padding: 2px 4px; border-radius: 3px; font-family: var(--font-mono);">$1</code>')
      .replace(/\\n/g, '<br>');
  }
  
  addMessageActions(messageEl, content) {
    const actionsDiv = document.createElement('div');
    actionsDiv.className = 'message-actions';
    
    const copyBtn = document.createElement('button');
    copyBtn.className = 'action-btn';
    copyBtn.textContent = '📋 Copy';
    copyBtn.onclick = () => this.copyToClipboard(content);
    
    actionsDiv.appendChild(copyBtn);
    messageEl.appendChild(actionsDiv);
  }
  
  async copyToClipboard(text) {
    try {
      await navigator.clipboard.writeText(text);
      // Could add a toast notification here
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }
  
  scrollToBottom() {
    this.elements.messages.scrollTop = this.elements.messages.scrollHeight;
  }
  
  exportConversation() {
    const timestamp = new Date().toISOString().split('T')[0];
    const content = this.conversation.map(msg => 
      `[${msg.timestamp.toLocaleTimeString()}] ${msg.role.toUpperCase()}: ${msg.content}`
    ).join('\\n\\n');
    
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `jnanaverse-chat-${timestamp}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  window.chatbot = new JnanaChatbot();
});
</script>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────
def load_translator(device_str):
    """Load facebook/nllb-200-distilled-600M for Sanskrit translation."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    print("[Translator] Loading facebook/nllb-200-distilled-600M …")
    name = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(name)
    dev = torch.device("cuda" if (device_str != "cpu" and torch.cuda.is_available()) else "cpu")
    mdl = mdl.to(dev).eval()
    print("[Translator] NLLB-200 ready.")
    return {"model": mdl, "tokenizer": tok, "device": dev}


def load_ner_pipeline(device_str):
    """Load NER pipeline for named entity recognition."""
    from transformers import pipeline
    print("[NER] Loading NER pipeline …")
    use_gpu = 0 if (device_str != "cpu" and torch.cuda.is_available()) else -1
    pipe = pipeline("ner", aggregation_strategy="simple", device=use_gpu)
    print("[NER] Ready.")
    return pipe


def load_sentiment_pipeline(device_str):
    """Load sentiment analysis pipeline."""
    from transformers import pipeline
    print("[Sentiment] Loading sentiment pipeline …")
    use_gpu = 0 if (device_str != "cpu" and torch.cuda.is_available()) else -1
    pipe = pipeline("sentiment-analysis", device=use_gpu)
    print("[Sentiment] Ready.")
    return pipe


def translate_to_sanskrit(text):
    """Translate English → Sanskrit using NLLB-200."""
    if TRANSLATOR is None:
        return "Translation model not loaded."
    try:
        tok = TRANSLATOR["tokenizer"]
        mdl = TRANSLATOR["model"]
        dev = TRANSLATOR["device"]
        
        tok.src_lang = "eng_Latn"
        inputs = tok(text, return_tensors="pt", padding=True).to(dev)
        
        with torch.no_grad():
            generated = mdl.generate(
                **inputs,
                forced_bos_token_id=tok.convert_tokens_to_ids("san_Deva"),
                num_beams=5,
                max_length=256,
            )
        result = tok.batch_decode(generated, skip_special_tokens=True)[0]
        
        # Add transliteration
        transliterated = transliterate_sanskrit(result)
        return f"{result}\\n\\n🔤 Transliteration: {transliterated}"
    except Exception as e:
        return f"Translation error: {e}"


def transliterate_sanskrit(devanagari_text):
    """Convert Devanagari to IAST (rough approximation)."""
    # This is a simplified mapping - in production, use libraries like indic-transliteration
    mapping = {
        'अ': 'a', 'आ': 'ā', 'इ': 'i', 'ई': 'ī', 'उ': 'u', 'ऊ': 'ū',
        'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
        'क': 'ka', 'ख': 'kha', 'ग': 'ga', 'घ': 'gha', 'ङ': 'ṅa',
        'च': 'ca', 'छ': 'cha', 'ज': 'ja', 'झ': 'jha', 'ञ': 'ña',
        'ट': 'ṭa', 'ठ': 'ṭha', 'ड': 'ḍa', 'ढ': 'ḍha', 'ण': 'ṇa',
        'त': 'ta', 'थ': 'tha', 'द': 'da', 'ध': 'dha', 'न': 'na',
        'प': 'pa', 'फ': 'pha', 'ब': 'ba', 'भ': 'bha', 'म': 'ma',
        'य': 'ya', 'र': 'ra', 'ल': 'la', 'व': 'va',
        'श': 'śa', 'ष': 'ṣa', 'स': 'sa', 'ह': 'ha',
        'ं': 'ṃ', 'ः': 'ḥ', '्': '', ' ': ' '
    }
    
    result = ""
    for char in devanagari_text:
        result += mapping.get(char, char)
    
    return result


def perform_ner(text):
    """Extract named entities from text."""
    if NER_PIPELINE is None:
        return "NER model not loaded."
    try:
        entities = NER_PIPELINE(text)
        if not entities:
            return "No named entities found."
        
        result = []
        for ent in entities:
            result.append(f"• {ent['word']} → {ent['entity_group']} ({ent['score']:.2f})")
        return "\\n".join(result)
    except Exception as e:
        return f"NER error: {e}"


def analyze_sentiment(text):
    """Analyze sentiment of text."""
    if SENTIMENT_PIPELINE is None:
        return "Sentiment model not loaded."
    try:
        result = SENTIMENT_PIPELINE(text)[0]
        label = result['label']
        confidence = result['score']
        
        emoji_map = {"POSITIVE": "😊", "NEGATIVE": "😞", "NEUTRAL": "😐"}
        emoji = emoji_map.get(label, "🤔")
        
        return f"{emoji} {label} (confidence: {confidence:.2f})"
    except Exception as e:
        return f"Sentiment error: {e}"


# ──────────────────────────────────────────────────────────────
# Flask routes
# ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")


@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True)
    task = body.get("task", "generation")
    model_type = body.get("model", "t5")
    session_id = body.get("session_id", "default")
    temperature = body.get("temperature", 0.8)
    max_tokens = body.get("max_tokens", 128)
    
    # Initialize session if not exists
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"history": []}
    
    try:
        if task == "generation":
            text = body.get("message", "").strip()
            if not text:
                return jsonify({"response": "(empty input)"})
            
            model = T5_MODEL if model_type == "t5" else CUSTOM_MODEL
            if model is None:
                return jsonify({"error": f"{model_type} model not available"}), 400
            
            if model_type == "t5":
                result = model.generate_text(text, device=str(DEVICE))
            else:
                # For custom model, tokenize and generate
                # This is a simplified example - you'd implement proper generation
                result = f"[Custom model response to: {text}]"
            
            return jsonify({"response": result or "(empty output)"})
        
        elif task == "translate":
            text = body.get("message", "").strip()
            if not text:
                return jsonify({"response": "(empty input)"})
            result = translate_to_sanskrit(text)
            return jsonify({"response": result})
        
        elif task == "summarize":
            text = body.get("message", "").strip()
            if not text:
                return jsonify({"response": "(empty input)"})
            prompt = f"summarize: {text}"
            result = T5_MODEL.generate_text(prompt, device=str(DEVICE))
            return jsonify({"response": result or "(summarization failed)"})
        
        elif task == "qa":
            text = body.get("message", "").strip()
            if not text:
                return jsonify({"response": "(empty input)"})
            # Expect format like "Q: What is X? Context: Y"
            if "Context:" not in text:
                return jsonify({"response": "Please provide format: Q: [question] Context: [context]"})
            result = T5_MODEL.generate_text(text, device=str(DEVICE))
            return jsonify({"response": result or "(no answer found)"})
        
        elif task == "sentiment":
            text = body.get("message", "").strip()
            if not text:
                return jsonify({"response": "(empty input)"})
            result = analyze_sentiment(text)
            return jsonify({"response": result})
        
        elif task == "ner":
            text = body.get("message", "").strip()
            if not text:
                return jsonify({"response": "(empty input)"})
            result = perform_ner(text)
            return jsonify({"response": result})
        
        elif task == "classify":
            text = body.get("message", "").strip()
            if not text:
                return jsonify({"response": "(empty input)"})
            
            enc = T5_MODEL.encode_text(text, device=str(DEVICE))
            with torch.no_grad():
                logits = T5_MODEL(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    task="classification",
                )
            probs = F.softmax(logits, dim=-1)[0]
            top_class = probs.argmax().item()
            confidence = probs[top_class].item()
            top5 = torch.topk(probs, min(5, len(probs)))
            
            lines = [f"Predicted class: {top_class} ({confidence:.1%} confidence)\\n"]
            for cls, p in zip(top5.indices.tolist(), top5.values.tolist()):
                bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
                lines.append(f"[{cls:>2}] {bar} {p:.1%}")
            
            return jsonify({"response": "\\n".join(lines)})
        
        elif task == "similarity":
            s1 = body.get("s1", "").strip()
            s2 = body.get("s2", "").strip()
            if not s1 or not s2:
                return jsonify({"response": "Two sentences required."})
            
            e1 = T5_MODEL.encode_text(s1, device=str(DEVICE))
            e2 = T5_MODEL.encode_text(s2, device=str(DEVICE))
            with torch.no_grad():
                score = T5_MODEL(
                    input_ids=(e1["input_ids"], e2["input_ids"]),
                    attention_mask=(e1["attention_mask"], e2["attention_mask"]),
                    task="similarity",
                ).item()
            
            bar_len = int((score + 1) / 2 * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            verdict = (
                "very similar" if score > 0.85 else
                "somewhat similar" if score > 0.5 else
                "loosely related" if score > 0.2 else
                "dissimilar"
            )
            
            result = (
                f'Sentence 1: "{s1}"\\n'
                f'Sentence 2: "{s2}"\\n\\n'
                f"Cosine similarity: {score:.4f}\\n"
                f"[{bar}]\\n"
                f"Verdict: {verdict}"
            )
            return jsonify({"response": result})
        
        else:
            return jsonify({"error": f"Unknown task: {task}"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="JnanaVerse Enhanced Chatbot")
    p.add_argument("--t5-model", default="t5-small", help="T5 model name")
    p.add_argument("--custom-vocab", default=5000, type=int, help="Custom model vocab size")
    p.add_argument("--classes", default=5, type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--port", default=5000, type=int)
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    
    DEVICE = get_device() if args.device == "auto" else torch.device(args.device)
    
    # Load T5 model
    print(f"[Main] Loading T5 model '{args.t5_model}' …")
    T5_MODEL = JnanaVerse(model_name=args.t5_model, num_classes=args.classes).to(DEVICE)
    T5_MODEL.eval()
    print(f"[Main] T5 ready ({count_parameters(T5_MODEL)} params)")
    
    # Load custom transformer (optional)
    try:
        print("[Main] Loading custom transformer …")
        CUSTOM_MODEL = CustomTransformerLM(
            vocab_size=args.custom_vocab,
            d_model=256,
            n_layers=4,
            n_heads=8,
            seq_len=128
        ).to(DEVICE)
        CUSTOM_MODEL.eval()
        print(f"[Main] Custom model ready ({count_parameters(CUSTOM_MODEL)} params)")
    except Exception as e:
        print(f"[Warning] Custom model not available: {e}")
        CUSTOM_MODEL = None
    
    # Load additional models
    TRANSLATOR = load_translator(str(DEVICE))
    NER_PIPELINE = load_ner_pipeline(str(DEVICE))
    SENTIMENT_PIPELINE = load_sentiment_pipeline(str(DEVICE))
    
    url = f"http://localhost:{args.port}"
    threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    print(f"\\n🚀 JnanaVerse Enhanced Chatbot starting at {url}\\n")
    
    app.run(host="0.0.0.0", port=args.port, debug=False)
