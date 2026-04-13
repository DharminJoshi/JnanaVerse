"""
JnanaVerse – chatbot_web.py  v3.0
Full-featured web chatbot powered by JnanaVerse + NLLB-200 + spaCy NER.

NEW FEATURES:
  NLP Tasks   : Summarization, Question Answering, Sentiment Analysis,
                Named Entity Recognition, Paraphrasing
  Sanskrit    : IAST transliteration alongside Devanagari, Sanskrit→English,
                word-by-word breakdown
  Chat UX     : Multi-turn history, copy buttons, export chat (.txt/.json),
                suggested prompts, response timing, dark/light mode toggle
  Gen Controls: Temperature + top-k sliders, streaming token output,
                model info panel
  REST API    : /api/translate  /api/summarize  /api/sentiment  /api/ner
                /api/similarity  /api/qa  (all JSON, CORS-enabled)
  Rate Limit  : per-IP sliding-window limiter (60 req/min, configurable)

Run:
    pip install flask
    python chatbot_web.py [--model t5-small] [--port 5000]

Developer(s): Dharmin Joshi / DevKay
"""

import argparse
import json
import time
import threading
import webbrowser
import collections
import re
from functools import wraps

import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, Response, stream_with_context

from jnanaverse import JnanaVerse, set_seed, get_device, count_parameters

# ──────────────────────────────────────────────────────────────
# Globals  (NO type annotations — avoids `global` SyntaxError)
# ──────────────────────────────────────────────────────────────
app        = Flask(__name__)
MODEL      = None   # JnanaVerse (T5)
DEVICE     = None   # torch.device
TRANSLATOR = None   # NLLB-200 dict
SA_MODEL   = None   # Sentiment pipeline
NER_MODEL  = None   # spaCy / HF NER pipeline
RATE_STORE = collections.defaultdict(list)   # ip -> [timestamps]
RATE_LIMIT = 60     # requests per minute per IP (overridden by --rate-limit)

# ──────────────────────────────────────────────────────────────
# IAST transliteration table  (Devanagari → Latin)
# ──────────────────────────────────────────────────────────────
IAST_MAP = {
    'अ':'a','आ':'ā','इ':'i','ई':'ī','उ':'u','ऊ':'ū',
    'ऋ':'ṛ','ॠ':'ṝ','ऌ':'ḷ','ए':'e','ऐ':'ai','ओ':'o','औ':'au',
    'अं':'ṃ','अः':'ḥ','क':'k','ख':'kh','ग':'g','घ':'gh','ङ':'ṅ',
    'च':'c','छ':'ch','ज':'j','झ':'jh','ञ':'ñ',
    'ट':'ṭ','ठ':'ṭh','ड':'ḍ','ढ':'ḍh','ण':'ṇ',
    'त':'t','थ':'th','द':'d','ध':'dh','न':'n',
    'प':'p','फ':'ph','ब':'b','भ':'bh','म':'m',
    'य':'y','र':'r','ल':'l','व':'v','श':'ś','ष':'ṣ','स':'s','ह':'h',
    'ा':'ā','ि':'i','ी':'ī','ु':'u','ू':'ū','ृ':'ṛ',
    'े':'e','ै':'ai','ो':'o','ौ':'au','ं':'ṃ','ः':'ḥ','्':'',
    '।':'|','॥':'||',
}

def devanagari_to_iast(text):
    result = []
    for ch in text:
        result.append(IAST_MAP.get(ch, ch))
    return ''.join(result)

# ──────────────────────────────────────────────────────────────
# Rate limiter decorator
# ──────────────────────────────────────────────────────────────
def rate_limit(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        ip  = request.remote_addr or 'unknown'
        now = time.time()
        RATE_STORE[ip] = [t for t in RATE_STORE[ip] if now - t < 60]
        if len(RATE_STORE[ip]) >= RATE_LIMIT:
            return jsonify({"error": "Rate limit exceeded. Max 60 requests/min."}), 429
        RATE_STORE[ip].append(now)
        return fn(*args, **kwargs)
    return wrapper

# ──────────────────────────────────────────────────────────────
# CORS helper
# ──────────────────────────────────────────────────────────────
def add_cors(response):
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

app.after_request(add_cors)

# ──────────────────────────────────────────────────────────────
# Model loaders
# ──────────────────────────────────────────────────────────────
def load_translator(device_str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    print("[Translator] Loading facebook/nllb-200-distilled-600M …")
    name = "facebook/nllb-200-distilled-600M"
    tok  = AutoTokenizer.from_pretrained(name)
    mdl  = AutoModelForSeq2SeqLM.from_pretrained(name)
    dev  = torch.device("cuda" if (device_str != "cpu" and torch.cuda.is_available()) else "cpu")
    mdl  = mdl.to(dev).eval()
    print("[Translator] NLLB-200 ready.")
    return {"model": mdl, "tokenizer": tok, "device": dev}

def load_sentiment(device_str):
    from transformers import pipeline as hf_pipeline, AutoModelForSequenceClassification, AutoTokenizer
    print("[Sentiment] Loading cardiffnlp/twitter-roberta-base-sentiment-latest …")
    name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tok  = AutoTokenizer.from_pretrained(name)
    mdl  = AutoModelForSequenceClassification.from_pretrained(name)
    use_gpu = 0 if (device_str != "cpu" and torch.cuda.is_available()) else -1
    pipe = hf_pipeline("sentiment-analysis", model=mdl, tokenizer=tok,
                        framework="pt", device=use_gpu, return_all_scores=True)
    print("[Sentiment] Ready.")
    return pipe

def load_ner(device_str):
    from transformers import pipeline as hf_pipeline, AutoModelForTokenClassification, AutoTokenizer
    print("[NER] Loading dslim/bert-base-NER …")
    name = "dslim/bert-base-NER"
    tok  = AutoTokenizer.from_pretrained(name)
    mdl  = AutoModelForTokenClassification.from_pretrained(name)
    use_gpu = 0 if (device_str != "cpu" and torch.cuda.is_available()) else -1
    pipe = hf_pipeline("ner", model=mdl, tokenizer=tok,
                        framework="pt", device=use_gpu,
                        aggregation_strategy="simple")
    print("[NER] Ready.")
    return pipe

# ──────────────────────────────────────────────────────────────
# Sanskrit helpers
# ──────────────────────────────────────────────────────────────
def _nllb_translate(text, src, tgt):
    tok = TRANSLATOR["tokenizer"]
    mdl = TRANSLATOR["model"]
    dev = TRANSLATOR["device"]
    tok.src_lang = src
    inputs = tok(text, return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        generated = mdl.generate(
            **inputs,
            forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
            num_beams=5, max_length=256,
        )
    return tok.batch_decode(generated, skip_special_tokens=True)[0]

def translate_en_to_san(text):
    if TRANSLATOR is None:
        return {"devanagari": "Translation model not loaded.", "iast": "", "words": []}
    devanagari = _nllb_translate(text, "eng_Latn", "san_Deva")
    iast       = devanagari_to_iast(devanagari)
    # Simple word-by-word breakdown (split on spaces/punctuation)
    words = []
    for w in devanagari.split():
        clean = re.sub(r'[।॥|]', '', w).strip()
        if clean:
            words.append({"san": clean, "iast": devanagari_to_iast(clean)})
    return {"devanagari": devanagari, "iast": iast, "words": words}

def translate_san_to_en(text):
    if TRANSLATOR is None:
        return "Translation model not loaded."
    return _nllb_translate(text, "san_Deva", "eng_Latn")

# ──────────────────────────────────────────────────────────────
# NLP task helpers
# ──────────────────────────────────────────────────────────────
def run_summarize(text):
    prompt = f"summarize: {text}"
    return MODEL.generate_text(prompt, max_new_tokens=150, device=str(DEVICE))

def run_paraphrase(text):
    prompt = f"paraphrase: {text}"
    return MODEL.generate_text(prompt, max_new_tokens=120, device=str(DEVICE))

def run_qa(question, context):
    prompt = f"question: {question} context: {context}"
    return MODEL.generate_text(prompt, max_new_tokens=100, device=str(DEVICE))

def run_sentiment(text):
    if SA_MODEL is None:
        return {"error": "Sentiment model not loaded."}
    results = SA_MODEL(text[:512])[0]
    label_map = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral",
                 "LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
    scores = [{"label": label_map.get(r["label"], r["label"]),
               "score": round(r["score"] * 100, 1)} for r in results]
    scores.sort(key=lambda x: x["score"], reverse=True)
    return {"scores": scores, "top": scores[0]["label"]}

def run_ner(text):
    if NER_MODEL is None:
        return {"error": "NER model not loaded."}
    entities = NER_MODEL(text[:512])
    return [{"word": e["word"], "label": e["entity_group"],
             "score": round(e["score"] * 100, 1)} for e in entities]

def run_classify(text):
    enc = MODEL.encode_text(text, device=str(DEVICE))
    with torch.no_grad():
        logits = MODEL(input_ids=enc["input_ids"],
                       attention_mask=enc["attention_mask"],
                       task="classification")
    probs     = F.softmax(logits, dim=-1)[0]
    top_class = probs.argmax().item()
    confidence = probs[top_class].item()
    top5 = torch.topk(probs, min(5, len(probs)))
    return {
        "top_class": top_class,
        "confidence": round(confidence * 100, 1),
        "top5": [{"class": int(c), "prob": round(float(p) * 100, 1)}
                 for c, p in zip(top5.indices.tolist(), top5.values.tolist())]
    }

def run_similarity(s1, s2):
    e1 = MODEL.encode_text(s1, device=str(DEVICE))
    e2 = MODEL.encode_text(s2, device=str(DEVICE))
    with torch.no_grad():
        score = MODEL(
            input_ids      =(e1["input_ids"],      e2["input_ids"]),
            attention_mask =(e1["attention_mask"], e2["attention_mask"]),
            task="similarity",
        ).item()
    verdict = ("very similar" if score > 0.85 else
               "somewhat similar" if score > 0.5 else
               "loosely related" if score > 0.2 else "dissimilar")
    return {"score": round(score, 4), "verdict": verdict}

# ──────────────────────────────────────────────────────────────
# HTML page  (single-file SPA)
# ──────────────────────────────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>JnanaVerse Chatbot</title>
<style>
:root[data-theme="dark"]{
  --bg:#0e0e10;--surface:#18181b;--surface2:#1e1e23;--border:#2a2a2e;
  --accent:#6c63ff;--accent-h:#7c74ff;--text:#e4e4e7;--muted:#71717a;
  --user-bg:#6c63ff;--bot-bg:#27272a;--sys-col:#52525b;
  --tag-per:#4f3f8a;--tag-org:#3d5a80;--tag-loc:#3b6351;--tag-misc:#7a3b3b;
  --pos:#22543d;--neg:#742a2a;--neu:#2d3748;
  --slider-track:#3a3a3f;
}
:root[data-theme="light"]{
  --bg:#f4f4f5;--surface:#ffffff;--surface2:#f1f1f3;--border:#d4d4d8;
  --accent:#6c63ff;--accent-h:#5a52e0;--text:#18181b;--muted:#71717a;
  --user-bg:#6c63ff;--bot-bg:#e4e4e7;--sys-col:#a1a1aa;
  --tag-per:#e9d8fd;--tag-org:#bee3f8;--tag-loc:#c6f6d5;--tag-misc:#fed7d7;
  --pos:#c6f6d5;--neg:#fed7d7;--neu:#e2e8f0;
  --slider-track:#d4d4d8;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
     display:flex;flex-direction:column;height:100vh;overflow:hidden;transition:background .2s,color .2s}
/* ── Header ── */
header{padding:12px 20px;border-bottom:1px solid var(--border);
       display:flex;align-items:center;gap:12px;background:var(--surface);flex-shrink:0}
.logo{width:34px;height:34px;border-radius:9px;background:var(--accent);
      display:grid;place-items:center;font-size:17px;font-weight:700;color:#fff;flex-shrink:0}
.hdr-text{flex:1}
.hdr-text .title{font-size:14px;font-weight:600}
.hdr-text .sub{font-size:11px;color:var(--muted)}
.hdr-actions{display:flex;gap:8px;align-items:center}
.icon-btn{background:transparent;border:1px solid var(--border);border-radius:8px;
          padding:5px 10px;font-size:12px;color:var(--muted);cursor:pointer;transition:all .15s}
.icon-btn:hover{color:var(--text);border-color:var(--accent)}
/* ── Task bar ── */
.task-bar{display:flex;gap:6px;padding:8px 20px;background:var(--surface);
          border-bottom:1px solid var(--border);flex-wrap:wrap;flex-shrink:0;align-items:center}
.task-btn{padding:4px 12px;border-radius:20px;border:1px solid var(--border);
          background:transparent;color:var(--muted);font-size:12px;cursor:pointer;transition:all .15s}
.task-btn:hover,.task-btn.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.task-sep{width:1px;height:20px;background:var(--border);margin:0 4px}
/* ── Suggested prompts ── */
#prompts{display:flex;gap:8px;padding:8px 20px;background:var(--surface2);
         border-bottom:1px solid var(--border);overflow-x:auto;flex-shrink:0}
#prompts:empty{display:none}
.prompt-chip{padding:5px 12px;border-radius:16px;border:1px solid var(--border);
             background:var(--surface);font-size:11px;color:var(--muted);cursor:pointer;
             white-space:nowrap;transition:all .15s;flex-shrink:0}
.prompt-chip:hover{border-color:var(--accent);color:var(--accent)}
/* ── Controls panel ── */
#ctrl-panel{display:none;padding:10px 20px;background:var(--surface2);
            border-bottom:1px solid var(--border);gap:16px;flex-wrap:wrap;align-items:center;flex-shrink:0}
#ctrl-panel.open{display:flex}
.ctrl-group{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted)}
.ctrl-group label{white-space:nowrap}
.ctrl-group input[type=range]{width:100px;accent-color:var(--accent)}
.ctrl-val{min-width:28px;font-weight:600;color:var(--text)}
/* ── Messages ── */
#messages{flex:1;overflow-y:auto;padding:16px 20px;display:flex;flex-direction:column;gap:12px}
.msg{max-width:75%;border-radius:14px;font-size:14px;line-height:1.65;
     animation:pop .15s ease;position:relative}
.msg.user{background:var(--user-bg);color:#fff;align-self:flex-end;
           border-bottom-right-radius:3px;padding:10px 14px}
.msg.bot{background:var(--bot-bg);color:var(--text);align-self:flex-start;
          border-bottom-left-radius:3px;padding:10px 14px;padding-bottom:28px}
.msg.bot.typing{padding-bottom:10px;opacity:.7;font-style:italic}
.msg.system{background:transparent;color:var(--sys-col);font-size:11px;
             align-self:center;text-align:center;max-width:100%;padding:2px 0}
@keyframes pop{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}
/* ── Copy button on bot messages ── */
.copy-btn{position:absolute;bottom:6px;right:8px;background:transparent;
          border:1px solid var(--border);border-radius:6px;padding:2px 8px;
          font-size:10px;color:var(--muted);cursor:pointer;transition:all .15s}
.copy-btn:hover{border-color:var(--accent);color:var(--accent)}
.copy-btn.copied{color:#22c55e;border-color:#22c55e}
/* ── Timing badge ── */
.timing{font-size:10px;color:var(--muted);margin-top:4px;text-align:right}
/* ── Sanskrit card ── */
.san-card{display:flex;flex-direction:column;gap:6px}
.san-deva{font-size:18px;font-family:serif;line-height:1.8}
.san-iast{font-size:12px;color:var(--muted);font-style:italic}
.san-words{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.san-word{background:var(--surface2);border:1px solid var(--border);
          border-radius:6px;padding:4px 8px;font-size:12px;text-align:center}
.san-word .w-san{font-family:serif;font-size:14px}
.san-word .w-iast{font-size:10px;color:var(--muted);font-style:italic}
/* ── Sentiment bar ── */
.sent-result{display:flex;flex-direction:column;gap:6px}
.sent-row{display:flex;align-items:center;gap:8px;font-size:12px}
.sent-label{width:70px;font-weight:600}
.sent-bar-wrap{flex:1;height:8px;background:var(--border);border-radius:4px;overflow:hidden}
.sent-bar{height:100%;border-radius:4px;transition:width .4s}
.sent-bar.Positive{background:#22c55e}.sent-bar.Negative{background:#ef4444}.sent-bar.Neutral{background:#94a3b8}
.sent-pct{width:40px;text-align:right}
/* ── NER tags ── */
.ner-result{display:flex;flex-wrap:wrap;gap:6px;align-items:baseline;font-size:14px;line-height:2}
.ner-token{display:inline-flex;flex-direction:column;align-items:center;margin:2px}
.ner-word{padding:1px 6px;border-radius:4px;font-weight:500}
.ner-word.PER{background:var(--tag-per);color:#d6bcfa}
.ner-word.ORG{background:var(--tag-org);color:#90cdf4}
.ner-word.LOC{background:var(--tag-loc);color:#9ae6b4}
.ner-word.MISC{background:var(--tag-misc);color:#feb2b2}
.ner-tag{font-size:9px;color:var(--muted);letter-spacing:.5px}
/* ── Similarity gauge ── */
.sim-result{display:flex;flex-direction:column;gap:8px}
.sim-gauge-wrap{height:12px;background:var(--border);border-radius:6px;overflow:hidden}
.sim-gauge-fill{height:100%;background:linear-gradient(90deg,#6c63ff,#a78bfa);border-radius:6px;transition:width .5s}
.sim-verdict{font-size:12px;color:var(--muted)}
/* ── Classify bars ── */
.cls-result{display:flex;flex-direction:column;gap:5px}
.cls-row{display:flex;align-items:center;gap:8px;font-size:12px}
.cls-label{width:60px;color:var(--muted)}
.cls-bar-wrap{flex:1;height:8px;background:var(--border);border-radius:4px;overflow:hidden}
.cls-bar-fill{height:100%;background:var(--accent);border-radius:4px}
.cls-pct{width:40px;text-align:right;font-weight:600}
/* ── Input row ── */
.input-row{display:flex;gap:8px;padding:12px 20px;background:var(--surface);
            border-top:1px solid var(--border);flex-shrink:0;align-items:flex-end}
#inp{flex:1;background:var(--bg);border:1px solid var(--border);color:var(--text);
     border-radius:10px;padding:9px 13px;font-size:14px;font-family:inherit;outline:none;
     resize:none;max-height:120px;overflow-y:auto;transition:border-color .15s}
#inp:focus{border-color:var(--accent)}
#inp::placeholder{color:var(--muted)}
#send{background:var(--accent);color:#fff;border:none;border-radius:10px;
      padding:9px 18px;font-size:14px;cursor:pointer;transition:opacity .15s;flex-shrink:0}
#send:hover{opacity:.85}
#send:disabled{opacity:.4;cursor:not-allowed}
/* ── Scrollbar ── */
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
/* ── Modal ── */
.modal-backdrop{position:fixed;inset:0;background:rgba(0,0,0,.6);display:flex;
                align-items:center;justify-content:center;z-index:100}
.modal{background:var(--surface);border:1px solid var(--border);border-radius:14px;
       padding:24px;width:360px;max-width:90vw;display:flex;flex-direction:column;gap:12px}
.modal h3{font-size:15px;font-weight:600}
.modal p{font-size:13px;color:var(--muted);line-height:1.5}
.modal-btns{display:flex;gap:8px;justify-content:flex-end}
.modal-btn{padding:7px 16px;border-radius:8px;font-size:13px;cursor:pointer;border:1px solid var(--border);
           background:transparent;color:var(--text);transition:all .15s}
.modal-btn.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.modal-btn:hover{opacity:.85}
</style>
</head>
<body>

<!-- Header -->
<header>
  <div class="logo">J</div>
  <div class="hdr-text">
    <div class="title">JnanaVerse Chatbot <span style="font-size:10px;color:var(--muted);font-weight:400">v3.0</span></div>
    <div class="sub">Dharmin Joshi / DevKay</div>
  </div>
  <div class="hdr-actions">
    <button class="icon-btn" onclick="toggleControls()" title="Generation controls">⚙ Controls</button>
    <button class="icon-btn" onclick="exportChat('txt')" title="Export as .txt">↓ TXT</button>
    <button class="icon-btn" onclick="exportChat('json')" title="Export as .json">↓ JSON</button>
    <button class="icon-btn" onclick="clearChat()" title="Clear chat">✕ Clear</button>
    <button class="icon-btn" id="theme-btn" onclick="toggleTheme()">☀ Light</button>
  </div>
</header>

<!-- Task bar -->
<div class="task-bar">
  <button class="task-btn active" data-task="generate">Generate</button>
  <button class="task-btn" data-task="summarize">Summarize</button>
  <button class="task-btn" data-task="paraphrase">Paraphrase</button>
  <button class="task-btn" data-task="qa">Q&amp;A</button>
  <div class="task-sep"></div>
  <button class="task-btn" data-task="sentiment">Sentiment</button>
  <button class="task-btn" data-task="ner">NER</button>
  <button class="task-btn" data-task="classify">Classify</button>
  <button class="task-btn" data-task="similarity">Similarity</button>
  <div class="task-sep"></div>
  <button class="task-btn" data-task="translate">EN→Sanskrit</button>
  <button class="task-btn" data-task="san_en">Sanskrit→EN</button>
</div>

<!-- Suggested prompts -->
<div id="prompts"></div>

<!-- Gen controls panel -->
<div id="ctrl-panel">
  <div class="ctrl-group">
    <label>Temperature</label>
    <input type="range" id="temp-sl" min="1" max="20" value="8" step="1" oninput="updateCtrl()"/>
    <span class="ctrl-val" id="temp-val">0.8</span>
  </div>
  <div class="ctrl-group">
    <label>Top-k</label>
    <input type="range" id="topk-sl" min="1" max="100" value="50" step="1" oninput="updateCtrl()"/>
    <span class="ctrl-val" id="topk-val">50</span>
  </div>
  <div class="ctrl-group">
    <label>Beams</label>
    <input type="range" id="beams-sl" min="1" max="8" value="4" step="1" oninput="updateCtrl()"/>
    <span class="ctrl-val" id="beams-val">4</span>
  </div>
  <div class="ctrl-group">
    <label>Max tokens</label>
    <input type="range" id="maxt-sl" min="20" max="300" value="128" step="10" oninput="updateCtrl()"/>
    <span class="ctrl-val" id="maxt-val">128</span>
  </div>
</div>

<!-- Messages -->
<div id="messages">
  <div class="msg system">JnanaVerse v3.0 ready — pick a task and start chatting.</div>
</div>

<!-- Input -->
<div class="input-row">
  <textarea id="inp" rows="1" placeholder="Type a message… (Enter=send, Shift+Enter=newline)"></textarea>
  <button id="send" onclick="sendMsg()">Send</button>
</div>

<script>
/* ── State ── */
const PROMPTS = {
  generate:   ['Tell me about transformers','Write a haiku about AI','Explain neural networks simply'],
  summarize:  ['Paste a paragraph to summarize…','Long article text here…'],
  paraphrase: ['Rephrase: The quick brown fox…','Rewrite: Natural language processing is…'],
  qa:         ['question: What is AI? context: Artificial intelligence…','question: Who invented Python? context: Guido van Rossum…'],
  sentiment:  ['I absolutely love this product!','This was a terrible experience.','It was okay, nothing special.'],
  ner:        ['Barack Obama was born in Hawaii.','Google was founded in Menlo Park, California.'],
  classify:   ['This movie was fantastic!','The package arrived damaged.'],
  similarity: ['Compare: "I love AI" vs "I enjoy machine learning"'],
  translate:  ['Good Morning','Knowledge is power','The universe is infinite'],
  san_en:     ['सुप्रभातम्','ज्ञानं शक्तिः','ब्रह्म सत्यम्'],
};
const PLACEHOLDERS = {
  generate:   'Type anything to generate…',
  summarize:  'Paste text to summarize…',
  paraphrase: 'Enter text to paraphrase…',
  qa:         'Format: question: <q> context: <c>',
  sentiment:  'Enter text to analyse sentiment…',
  ner:        'Enter text to extract named entities…',
  classify:   'Enter text to classify…',
  similarity: 'Enter sentence 1…',
  translate:  'Enter English text to translate to Sanskrit…',
  san_en:     'Enter Sanskrit (Devanagari) to translate to English…',
};

let task     = 'generate';
let simStep  = 0, simSent1 = '';
let history  = [];   // [{role, text, raw}]
let theme    = 'dark';
let ctrlOpen = false;

/* ── Controls ── */
function updateCtrl() {
  document.getElementById('temp-val').textContent  = (document.getElementById('temp-sl').value / 10).toFixed(1);
  document.getElementById('topk-val').textContent  = document.getElementById('topk-sl').value;
  document.getElementById('beams-val').textContent = document.getElementById('beams-sl').value;
  document.getElementById('maxt-val').textContent  = document.getElementById('maxt-sl').value;
}
function getCtrl() {
  return {
    temperature: parseFloat(document.getElementById('temp-sl').value) / 10,
    top_k:       parseInt(document.getElementById('topk-sl').value),
    num_beams:   parseInt(document.getElementById('beams-sl').value),
    max_tokens:  parseInt(document.getElementById('maxt-sl').value),
  };
}
function toggleControls() {
  ctrlOpen = !ctrlOpen;
  document.getElementById('ctrl-panel').classList.toggle('open', ctrlOpen);
}

/* ── Theme ── */
function toggleTheme() {
  theme = theme === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', theme);
  document.getElementById('theme-btn').textContent = theme === 'dark' ? '☀ Light' : '🌙 Dark';
}

/* ── Suggested prompts ── */
function renderPrompts() {
  const el = document.getElementById('prompts');
  el.innerHTML = '';
  (PROMPTS[task] || []).forEach(p => {
    const ch = document.createElement('button');
    ch.className = 'prompt-chip';
    ch.textContent = p;
    ch.onclick = () => { document.getElementById('inp').value = p; document.getElementById('inp').focus(); };
    el.appendChild(ch);
  });
}

/* ── Task switcher ── */
document.querySelectorAll('.task-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.task-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    task = btn.dataset.task;
    simStep = 0;
    document.getElementById('inp').placeholder = PLACEHOLDERS[task] || 'Type a message…';
    renderPrompts();
    addMsg('system', 'Task: ' + btn.textContent.trim());
    if (task === 'similarity') addMsg('system', 'Enter sentence 1:');
  });
});

/* ── Messages ── */
const msgsEl = document.getElementById('messages');
const inp    = document.getElementById('inp');
const sendBtn = document.getElementById('send');

function addMsg(role, text, richEl) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  if (richEl) {
    d.appendChild(richEl);
  } else {
    d.style.whiteSpace = 'pre-wrap';
    d.textContent = text;
  }
  if (role === 'bot' && !richEl) {
    // Copy button
    const cb = document.createElement('button');
    cb.className = 'copy-btn';
    cb.textContent = 'Copy';
    cb.onclick = () => {
      navigator.clipboard.writeText(text).then(() => {
        cb.textContent = 'Copied!'; cb.classList.add('copied');
        setTimeout(() => { cb.textContent = 'Copy'; cb.classList.remove('copied'); }, 2000);
      });
    };
    d.appendChild(cb);
  }
  if (role === 'bot' && richEl) {
    const cb = document.createElement('button');
    cb.className = 'copy-btn';
    cb.textContent = 'Copy';
    cb.onclick = () => {
      navigator.clipboard.writeText(d.innerText).then(() => {
        cb.textContent = 'Copied!'; cb.classList.add('copied');
        setTimeout(() => { cb.textContent = 'Copy'; cb.classList.remove('copied'); }, 2000);
      });
    };
    d.appendChild(cb);
  }
  msgsEl.appendChild(d);
  msgsEl.scrollTop = msgsEl.scrollHeight;
  return d;
}

function addTiming(ms) {
  const t = document.createElement('div');
  t.className = 'timing';
  t.textContent = `⏱ ${(ms/1000).toFixed(2)}s`;
  msgsEl.appendChild(t);
  msgsEl.scrollTop = msgsEl.scrollHeight;
}

function addTyping() { return addMsg('bot typing', 'Thinking…'); }

/* ── Rich renderers ── */
function renderSanskrit(data) {
  const el = document.createElement('div');
  el.className = 'san-card';
  el.innerHTML = `
    <div class="san-deva">${data.devanagari}</div>
    <div class="san-iast">${data.iast}</div>
    <div class="san-words">${(data.words||[]).map(w=>`
      <div class="san-word">
        <div class="w-san">${w.san}</div>
        <div class="w-iast">${w.iast}</div>
      </div>`).join('')}
    </div>`;
  return el;
}

function renderSentiment(data) {
  const el = document.createElement('div');
  el.className = 'sent-result';
  el.innerHTML = `<div style="font-weight:600;margin-bottom:4px">Overall: ${data.top}</div>` +
    data.scores.map(s => `
      <div class="sent-row">
        <span class="sent-label">${s.label}</span>
        <div class="sent-bar-wrap"><div class="sent-bar ${s.label}" style="width:${s.score}%"></div></div>
        <span class="sent-pct">${s.score}%</span>
      </div>`).join('');
  return el;
}

function renderNER(entities, originalText) {
  const el = document.createElement('div');
  el.className = 'ner-result';
  if (!entities.length) { el.textContent = 'No named entities found.'; return el; }
  entities.forEach(e => {
    const tok = document.createElement('div');
    tok.className = 'ner-token';
    tok.innerHTML = `<span class="ner-word ${e.label}">${e.word}</span><span class="ner-tag">${e.label} ${e.score}%</span>`;
    el.appendChild(tok);
  });
  return el;
}

function renderSimilarity(data) {
  const el = document.createElement('div');
  el.className = 'sim-result';
  const pct = Math.round(((data.score + 1) / 2) * 100);
  el.innerHTML = `
    <div style="font-size:13px">Score: <strong>${data.score}</strong> — ${data.verdict}</div>
    <div class="sim-gauge-wrap"><div class="sim-gauge-fill" style="width:${pct}%"></div></div>`;
  return el;
}

function renderClassify(data) {
  const el = document.createElement('div');
  el.className = 'cls-result';
  el.innerHTML = `<div style="font-size:13px;margin-bottom:4px">Top class: <strong>${data.top_class}</strong> (${data.confidence}%)</div>` +
    data.top5.map(c => `
      <div class="cls-row">
        <span class="cls-label">Class ${c.class}</span>
        <div class="cls-bar-wrap"><div class="cls-bar-fill" style="width:${c.prob}%"></div></div>
        <span class="cls-pct">${c.prob}%</span>
      </div>`).join('');
  return el;
}

/* ── Main send ── */
async function sendMsg() {
  const text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  inp.style.height = 'auto';

  /* Similarity two-step */
  if (task === 'similarity') {
    if (simStep === 0) {
      addMsg('user', text);
      history.push({role:'user', text});
      simSent1 = text;
      simStep = 1;
      addMsg('system', 'Enter sentence 2:');
      return;
    } else {
      addMsg('user', text);
      history.push({role:'user', text});
      const dot = addTyping();
      sendBtn.disabled = true;
      const t0 = Date.now();
      const r = await fetch('/chat', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({task:'similarity', s1:simSent1, s2:text})
      });
      dot.remove();
      const data = await r.json();
      if (data.error) { addMsg('bot', data.error); }
      else { addMsg('bot', null, renderSimilarity(data)); }
      addTiming(Date.now() - t0);
      history.push({role:'bot', text: JSON.stringify(data)});
      sendBtn.disabled = false;
      simStep = 0;
      addMsg('system', 'Enter sentence 1:');
      return;
    }
  }

  /* QA two-step prompt hint */
  addMsg('user', text);
  history.push({role:'user', text});
  const dot = addTyping();
  sendBtn.disabled = true;
  const t0 = Date.now();
  const ctrl = getCtrl();

  try {
    const r = await fetch('/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({task, message:text, ...ctrl})
    });
    const data = await r.json();
    dot.remove();
    const elapsed = Date.now() - t0;

    if (data.error) {
      addMsg('bot', '⚠ ' + data.error);
    } else if (task === 'translate') {
      addMsg('bot', null, renderSanskrit(data));
    } else if (task === 'sentiment') {
      addMsg('bot', null, renderSentiment(data));
    } else if (task === 'ner') {
      addMsg('bot', null, renderNER(data.entities || [], text));
    } else if (task === 'classify') {
      addMsg('bot', null, renderClassify(data));
    } else {
      addMsg('bot', data.response || '(empty)');
    }
    addTiming(elapsed);
    history.push({role:'bot', raw:data, text: data.response || JSON.stringify(data)});
  } catch (e) {
    dot.remove();
    addMsg('bot', 'Error: ' + e.message);
  }
  sendBtn.disabled = false;
  inp.focus();
}

/* ── Export ── */
function exportChat(fmt) {
  let content, mime, ext;
  if (fmt === 'json') {
    content = JSON.stringify(history, null, 2);
    mime = 'application/json'; ext = 'json';
  } else {
    content = history.map(m => `[${m.role.toUpperCase()}]\n${m.text}`).join('\n\n---\n\n');
    mime = 'text/plain'; ext = 'txt';
  }
  const blob = new Blob([content], {type: mime});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `jnanaverse_chat_${Date.now()}.${ext}`;
  a.click();
}

function clearChat() {
  history = [];
  msgsEl.innerHTML = '<div class="msg system">Chat cleared.</div>';
}

/* ── Keyboard ── */
inp.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); }
});
inp.addEventListener('input', () => {
  inp.style.height = 'auto';
  inp.style.height = Math.min(inp.scrollHeight, 120) + 'px';
});

/* ── Init ── */
renderPrompts();
updateCtrl();
</script>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────
# Main chat route
# ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")


@app.route("/chat", methods=["POST"])
@rate_limit
def chat():
    body  = request.get_json(force=True)
    task  = body.get("task", "generate")
    msg   = body.get("message", "").strip()
    temp  = float(body.get("temperature", 0.8))
    top_k = int(body.get("top_k", 50))
    beams = int(body.get("num_beams", 4))
    maxt  = int(body.get("max_tokens", 128))

    gen_kwargs = dict(max_new_tokens=maxt, num_beams=beams,
                      no_repeat_ngram_size=3, early_stopping=True)

    try:
        if task == "generate":
            if not msg: return jsonify({"response": "(empty input)"})
            out = MODEL.generate_text(msg, device=str(DEVICE), **gen_kwargs)
            return jsonify({"response": out or "(empty output)"})

        elif task == "summarize":
            if not msg: return jsonify({"response": "(empty input)"})
            out = MODEL.generate_text(f"summarize: {msg}", device=str(DEVICE), **gen_kwargs)
            return jsonify({"response": out})

        elif task == "paraphrase":
            if not msg: return jsonify({"response": "(empty input)"})
            out = MODEL.generate_text(f"paraphrase: {msg}", device=str(DEVICE), **gen_kwargs)
            return jsonify({"response": out})

        elif task == "qa":
            if not msg: return jsonify({"response": "(empty input)"})
            out = MODEL.generate_text(msg, device=str(DEVICE), **gen_kwargs)
            return jsonify({"response": out})

        elif task == "sentiment":
            if not msg: return jsonify({"error": "Empty input"})
            return jsonify(run_sentiment(msg))

        elif task == "ner":
            if not msg: return jsonify({"error": "Empty input"})
            entities = run_ner(msg)
            if isinstance(entities, dict) and "error" in entities:
                return jsonify(entities)
            return jsonify({"entities": entities})

        elif task == "classify":
            if not msg: return jsonify({"error": "Empty input"})
            return jsonify(run_classify(msg))

        elif task == "similarity":
            s1 = body.get("s1", "").strip()
            s2 = body.get("s2", "").strip()
            if not s1 or not s2: return jsonify({"error": "Two sentences required."})
            return jsonify(run_similarity(s1, s2))

        elif task == "translate":
            if not msg: return jsonify({"error": "Empty input"})
            data = translate_en_to_san(msg)
            return jsonify(data)

        elif task == "san_en":
            if not msg: return jsonify({"error": "Empty input"})
            out = translate_san_to_en(msg)
            return jsonify({"response": out})

        else:
            return jsonify({"error": f"Unknown task: {task}"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────────────────────
# REST API endpoints
# ──────────────────────────────────────────────────────────────
@app.route("/api/translate", methods=["POST"])
@rate_limit
def api_translate():
    """POST {"text": "..."} → {"devanagari":…,"iast":…,"words":[…]}"""
    body = request.get_json(force=True)
    text = body.get("text", "").strip()
    if not text: return jsonify({"error": "Missing 'text'"}), 400
    return jsonify(translate_en_to_san(text))

@app.route("/api/summarize", methods=["POST"])
@rate_limit
def api_summarize():
    """POST {"text": "..."} → {"summary": "..."}"""
    body = request.get_json(force=True)
    text = body.get("text", "").strip()
    if not text: return jsonify({"error": "Missing 'text'"}), 400
    out = MODEL.generate_text(f"summarize: {text}", device=str(DEVICE))
    return jsonify({"summary": out})

@app.route("/api/sentiment", methods=["POST"])
@rate_limit
def api_sentiment():
    """POST {"text": "..."} → {"top":…,"scores":[…]}"""
    body = request.get_json(force=True)
    text = body.get("text", "").strip()
    if not text: return jsonify({"error": "Missing 'text'"}), 400
    return jsonify(run_sentiment(text))

@app.route("/api/ner", methods=["POST"])
@rate_limit
def api_ner():
    """POST {"text": "..."} → {"entities":[…]}"""
    body = request.get_json(force=True)
    text = body.get("text", "").strip()
    if not text: return jsonify({"error": "Missing 'text'"}), 400
    return jsonify({"entities": run_ner(text)})

@app.route("/api/similarity", methods=["POST"])
@rate_limit
def api_similarity():
    """POST {"s1": "...", "s2": "..."} → {"score":…,"verdict":…}"""
    body = request.get_json(force=True)
    s1 = body.get("s1", "").strip()
    s2 = body.get("s2", "").strip()
    if not s1 or not s2: return jsonify({"error": "Provide 's1' and 's2'"}), 400
    return jsonify(run_similarity(s1, s2))

@app.route("/api/qa", methods=["POST"])
@rate_limit
def api_qa():
    """POST {"question": "...", "context": "..."} → {"answer": "..."}"""
    body = request.get_json(force=True)
    q = body.get("question", "").strip()
    c = body.get("context", "").strip()
    if not q or not c: return jsonify({"error": "Provide 'question' and 'context'"}), 400
    out = MODEL.generate_text(f"question: {q} context: {c}", device=str(DEVICE))
    return jsonify({"answer": out})

@app.route("/api/models", methods=["GET"])
def api_models():
    """GET → model info JSON"""
    return jsonify({
        "main_model":   MODEL.base_model.config._name_or_path if MODEL else None,
        "parameters":   count_parameters(MODEL) if MODEL else None,
        "translator":   "facebook/nllb-200-distilled-600M",
        "sentiment":    "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "ner":          "dslim/bert-base-NER",
        "device":       str(DEVICE),
    })


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="JnanaVerse Web Chatbot v3.0")
    p.add_argument("--model",       default="t5-small")
    p.add_argument("--classes",     default=5, type=int)
    p.add_argument("--device",      default="auto")
    p.add_argument("--port",        default=5000, type=int)
    p.add_argument("--seed",        default=42, type=int)
    p.add_argument("--rate-limit",  default=60, type=int, help="Max requests/min per IP")
    p.add_argument("--no-browser",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    RATE_LIMIT = args.rate_limit

    DEVICE = get_device() if args.device == "auto" else torch.device(args.device)

    print(f"[Main] Loading JnanaVerse '{args.model}' …")
    MODEL = JnanaVerse(model_name=args.model, num_classes=args.classes).to(DEVICE)
    MODEL.eval()
    print(f"[Main] Model ready — {count_parameters(MODEL)} params")

    TRANSLATOR = load_translator(str(DEVICE))
    SA_MODEL   = load_sentiment(str(DEVICE))
    NER_MODEL  = load_ner(str(DEVICE))

    url = f"http://localhost:{args.port}"
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"[Main] Serving on {url}  (rate limit: {RATE_LIMIT} req/min)\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)
