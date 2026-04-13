# JnanaVerse v2.0 Enhanced

**Advanced Multi-Task NLP Framework with Comprehensive Chatbot Features**  
*Developer(s): Dharmin Joshi / DevKay*

---

## 🚀 New Enhanced Features

### **Multi-Modal Chatbot Interfaces**
- **Web Chatbot** — Modern dark/light theme UI with streaming responses
- **CLI Chatbot** — Rich terminal interface with colors and history
- **Model Switching** — Toggle between T5 and Custom Transformer models live

### **Expanded NLP Tasks**
- **Text Generation** — Creative writing, completion, dialogue
- **Sanskrit Translation** — English → Sanskrit with IAST transliteration  
- **Text Summarization** — Condense long articles and documents
- **Question Answering** — Context-based Q&A system
- **Sentiment Analysis** — Emotion detection with confidence scores
- **Named Entity Recognition** — Extract people, places, organizations
- **Text Classification** — Multi-class text categorization
- **Semantic Similarity** — Compare sentence meanings

### **Advanced UI Features**
- **Streaming Output** — Real-time token-by-token generation
- **Conversation History** — Session memory with timestamps
- **Export Functionality** — Save chats as `.txt` files
- **Copy Buttons** — One-click response copying
- **Parameter Tuning** — Adjust temperature, max tokens, beams
- **Dark/Light Themes** — Toggle UI appearance
- **Responsive Design** — Works on desktop and mobile

---

## 📁 Project Structure

```
jnanaverse/
├── jnanaverse/                    # Core framework package
│   ├── model.py                   # JnanaVerse (T5) + CustomTransformerLM  
│   ├── tokenizer.py               # BPE tokenizer utilities
│   ├── dataset.py                 # Dataset classes for training
│   ├── trainer.py                 # Training loops + optimizers
│   └── utils.py                   # Checkpointing, metrics, helpers
├── enhanced_chatbot.py            # 🌟 Advanced web chatbot
├── enhanced_cli_chatbot.py        # 🌟 Advanced CLI chatbot  
├── chatbot_web.py                 # Basic web chatbot
├── chatbot.py                     # Basic CLI chatbot
├── demo_jnanaverse.py             # Framework demo
├── train_custom_lm.py             # Train custom Transformer
├── requirements_enhanced.txt       # 🌟 Enhanced dependencies
├── requirements.txt               # Basic dependencies
└── README.md                      # This file
```

---

## 🛠 Installation & Setup

### 1. Install Dependencies
```bash
# Enhanced features (recommended)
pip install -r requirements_enhanced.txt

# Or basic features only
pip install -r requirements.txt
```

### 2. Run Enhanced Chatbots

**Web Interface (Recommended)**
```bash
python enhanced_chatbot.py
# Opens http://localhost:5000 with full-featured web UI
# Features: streaming, history, export, themes, model switching
```

**Terminal Interface**
```bash
python enhanced_cli_chatbot.py  
# Rich CLI with colors, history, all NLP tasks
# Commands: /help, /export, /model, /settings
```

### 3. Basic Chatbots
```bash
# Basic web chatbot
python chatbot_web.py

# Basic CLI chatbot  
python chatbot.py
```

---

## 💬 Enhanced Chatbot Usage

### **Available Tasks**
| Task | Command | Description | Example |
|------|---------|-------------|---------|
| **Generation** | `/generate` | Creative text generation | Continue this story... |
| **Translation** | `/translate` | English → Sanskrit + transliteration | Hello world |
| **Summarization** | `/summarize` | Condense long text | *[paste article]* |
| **Q&A** | `/qa` | Context-based questions | Q: What is AI? Context: ... |
| **Sentiment** | `/sentiment` | Emotion analysis | I love this product! |
| **NER** | `/ner` | Extract entities | Apple Inc. was founded by Steve Jobs. |
| **Classification** | `/classify` | Categorize text | This movie is amazing... |
| **Similarity** | `/similarity` | Compare sentences | [prompts for 2 sentences] |

### **System Commands**
| Command | Function |
|---------|----------|
| `/model` | Switch between T5 ↔ Custom Transformer |
| `/settings` | View/adjust generation parameters |
| `/history` | Show conversation timeline |
| `/export` | Save chat to file |
| `/clear` | Clear conversation |
| `/help` | Show all commands |

### **Web UI Features**
- **🎨 Dark/Light Mode** — Toggle with moon/sun button
- **⚙️ Model Selection** — Dropdown to switch models
- **📊 Real-time Settings** — Temperature and token sliders
- **💾 Auto-save History** — Persistent across sessions  
- **📱 Mobile Responsive** — Sidebar collapse on mobile
- **🔄 Streaming Text** — See responses generate live

---

## 🔧 Configuration

### **Generation Parameters**
```bash
# In CLI chatbot
/set temperature 0.8        # Creativity (0.1-2.0)
/set max_tokens 128         # Response length (16-512)  
/set beams 4               # Quality vs speed (1-10)
/set top_k 50              # Diversity (1-100)
/set top_p 0.9             # Nucleus sampling (0.1-1.0)
```

### **Model Options**
```bash
# Launch with different models
python enhanced_chatbot.py --t5-model t5-base --port 8080
python enhanced_cli_chatbot.py --t5-model t5-large --device cuda
```

---

## 📚 Framework Usage

The enhanced chatbots are built on the same core framework:

```python
from jnanaverse import JnanaVerse, set_seed, get_device

# Initialize
set_seed(42)
device = get_device()
model = JnanaVerse(model_name="t5-small").to(device)

# Multi-task usage
result = model.generate_text("translate English to Sanskrit: Hello")
classification = model(inputs, task="classification")
similarity = model((input1, input2), task="similarity")
```

---

## 🌍 Language Support

**Sanskrit Translation Features:**
- **Source:** English text input
- **Target:** Devanagari script output  
- **Transliteration:** IAST notation for pronunciation
- **Model:** `facebook/nllb-200-distilled-600M` (2.4GB, unrestricted)

**Example Output:**
```
English: "Good morning"
Sanskrit: सुप्रभातम्
Transliteration: suprabhatam
```

---

## 🎯 Use Cases

### **Research & Development**
- **Multi-task NLP experimentation** with unified interface
- **Sanskrit digitization** for cultural preservation projects  
- **Educational tools** for language learning
- **Content analysis** pipelines (sentiment + NER + classification)

### **Production Applications**  
- **Customer support chatbots** with context understanding
- **Content moderation** systems with sentiment analysis
- **Document processing** with summarization + entity extraction
- **Cross-lingual applications** with Sanskrit support

---

## 🏗 Architecture

**Core Models:**
- **T5 (HuggingFace)** — Seq2seq for generation, summarization, QA, translation
- **Custom Transformer** — Decoder-only model trainable from scratch  
- **NLLB-200** — 200-language translation including Sanskrit
- **RoBERTa** — NER and sentiment analysis (via transformers pipelines)

**Key Features:**
- **Modular design** — swap models without breaking interface
- **Adapter support** — lightweight fine-tuning via `adapters` package
- **Memory efficiency** — smart model loading and device management
- **Extensible** — easy to add new tasks and models

---

## 📄 License

MIT [LICENSE](LICENSE) — use it, modify it, ship it.

---

---

**🧠 Experience the future of open-source NLP with JnanaVerse Enhanced!**
