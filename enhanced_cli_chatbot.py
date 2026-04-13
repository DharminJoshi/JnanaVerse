"""
JnanaVerse – enhanced_cli_chatbot.py
Advanced CLI chatbot with comprehensive NLP features:
- All original tasks plus summarization, QA, sentiment, NER
- Conversation history, export functionality
- Model switching, parameter tuning
- Sanskrit transliteration

Developer(s): Dharmin Joshi / DevKay
"""

import argparse
import json
import datetime
import sys
import os
import torch
import torch.nn.functional as F

from jnanaverse import JnanaVerse, CustomTransformerLM, set_seed, get_device, count_parameters

# ──────────────────────────────────────────────────────────────
# ANSI colours with fallback
# ──────────────────────────────────────────────────────────────
try:
    import colorama
    colorama.init()
    RESET = colorama.Style.RESET_ALL
    BOLD = colorama.Style.BRIGHT
    DIM = colorama.Style.DIM
    CYAN = colorama.Fore.CYAN
    GREEN = colorama.Fore.GREEN
    YELLOW = colorama.Fore.YELLOW
    RED = colorama.Fore.RED
    MAGENTA = colorama.Fore.MAGENTA
    BLUE = colorama.Fore.BLUE
except ImportError:
    RESET = BOLD = DIM = CYAN = GREEN = YELLOW = RED = MAGENTA = BLUE = ""

BANNER = f"""{CYAN}{BOLD}
╔══════════════════════════════════════════════════════════╗
║        🧠 JnanaVerse Enhanced Chatbot v2.0              ║
║           Developer(s): Dharmin Joshi / DevKay           ║
║                                                          ║
║  ✨ Features: Multi-task NLP, History, Export, Models   ║
╚══════════════════════════════════════════════════════════╝
{RESET}

{GREEN}Available Commands:{RESET}
{BLUE}/generate{RESET}    Text generation                {BLUE}/summarize{RESET}   Summarize text
{BLUE}/translate{RESET}   English → Sanskrit             {BLUE}/qa{RESET}         Question answering  
{BLUE}/sentiment{RESET}   Sentiment analysis             {BLUE}/ner{RESET}        Named entity recognition
{BLUE}/classify{RESET}    Text classification            {BLUE}/similarity{RESET} Sentence similarity

{GREEN}System Commands:{RESET}
{BLUE}/model{RESET}       Switch models (T5/Custom)      {BLUE}/history{RESET}     Show conversation history
{BLUE}/export{RESET}      Export chat to file           {BLUE}/settings{RESET}    Adjust parameters
{BLUE}/clear{RESET}       Clear conversation             {BLUE}/help{RESET}        Show this help
{BLUE}/exit{RESET}        Quit chatbot

Type {YELLOW}/help{RESET} anytime or press {YELLOW}Ctrl+C{RESET} to quit.
"""

TASK_LABELS = {
    "generation": "💬 Text Generation",
    "translate": "🌐 English → Sanskrit", 
    "summarize": "📝 Text Summarization",
    "qa": "❓ Question Answering",
    "sentiment": "😊 Sentiment Analysis",
    "ner": "🏷️ Named Entity Recognition", 
    "classify": "📊 Text Classification",
    "similarity": "🔗 Sentence Similarity",
}


# ──────────────────────────────────────────────────────────────
# Enhanced ChatBot Class
# ──────────────────────────────────────────────────────────────
class EnhancedJnanaChat:
    def __init__(self, t5_model="t5-small", custom_vocab=5000, num_classes=5, device_str="auto"):
        self.device = get_device() if device_str == "auto" else torch.device(device_str)
        self.task = "generation"
        self.current_model = "t5"  # "t5" or "custom"
        self.conversation_history = []
        
        # Generation settings
        self.settings = {
            "temperature": 0.8,
            "max_new_tokens": 128,
            "num_beams": 4,
            "top_k": 50,
            "top_p": 0.9
        }
        
        # Load T5 model
        print(f"\\n{CYAN}Loading T5 model '{t5_model}' …{RESET}")
        self.t5_model = JnanaVerse(model_name=t5_model, num_classes=num_classes).to(self.device)
        self.t5_model.eval()
        print(f"{GREEN}T5 ready. Parameters: {count_parameters(self.t5_model)}{RESET}")
        
        # Load custom transformer
        try:
            print(f"{CYAN}Loading custom transformer …{RESET}")
            self.custom_model = CustomTransformerLM(
                vocab_size=custom_vocab,
                d_model=256,
                n_layers=4,
                n_heads=8,
                seq_len=128
            ).to(self.device)
            self.custom_model.eval()
            print(f"{GREEN}Custom model ready. Parameters: {count_parameters(self.custom_model)}{RESET}")
        except Exception as e:
            print(f"{YELLOW}[Warning] Custom model not available: {e}{RESET}")
            self.custom_model = None
        
        # Load additional models
        self.translator = self._load_translator()
        self.ner_pipeline = self._load_ner_pipeline()
        self.sentiment_pipeline = self._load_sentiment_pipeline()

    def _load_translator(self):
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            print(f"{CYAN}Loading facebook/nllb-200-distilled-600M for Sanskrit …{RESET}")
            name = "facebook/nllb-200-distilled-600M"
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(name)
            mdl = mdl.to(self.device).eval()
            print(f"{GREEN}NLLB-200 ready.{RESET}")
            return {"model": mdl, "tokenizer": tok}
        except Exception as e:
            print(f"{YELLOW}[Warning] Could not load translator: {e}{RESET}")
            return None

    def _load_ner_pipeline(self):
        try:
            from transformers import pipeline
            print(f"{CYAN}Loading NER pipeline …{RESET}")
            use_gpu = 0 if (str(self.device) != "cpu" and torch.cuda.is_available()) else -1
            pipe = pipeline("ner", aggregation_strategy="simple", device=use_gpu)
            print(f"{GREEN}NER ready.{RESET}")
            return pipe
        except Exception as e:
            print(f"{YELLOW}[Warning] Could not load NER: {e}{RESET}")
            return None

    def _load_sentiment_pipeline(self):
        try:
            from transformers import pipeline
            print(f"{CYAN}Loading sentiment pipeline …{RESET}")
            use_gpu = 0 if (str(self.device) != "cpu" and torch.cuda.is_available()) else -1
            pipe = pipeline("sentiment-analysis", device=use_gpu)
            print(f"{GREEN}Sentiment analysis ready.{RESET}")
            return pipe
        except Exception as e:
            print(f"{YELLOW}[Warning] Could not load sentiment: {e}{RESET}")
            return None

    # ── Core response routing ────────────────────────────────────
    def respond(self, user_input: str) -> str:
        # Store in conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.datetime.now(),
            "task": self.task,
            "model": self.current_model
        })
        
        try:
            if self.task == "generation":
                result = self._generate(user_input)
            elif self.task == "translate":
                result = self._translate_sanskrit(user_input)
            elif self.task == "summarize":
                result = self._summarize(user_input)
            elif self.task == "qa":
                result = self._question_answering(user_input)
            elif self.task == "sentiment":
                result = self._analyze_sentiment(user_input)
            elif self.task == "ner":
                result = self._extract_entities(user_input)
            elif self.task == "classify":
                result = self._classify(user_input)
            else:
                result = f"{RED}Unknown task '{self.task}'. Type /generate to reset.{RESET}"
        except Exception as e:
            result = f"{RED}Error: {e}{RESET}"
        
        # Store bot response
        self.conversation_history.append({
            "role": "assistant", 
            "content": result,
            "timestamp": datetime.datetime.now(),
            "task": self.task,
            "model": self.current_model
        })
        
        return result

    def _generate(self, text: str) -> str:
        model = self.t5_model if self.current_model == "t5" else self.custom_model
        if model is None:
            return f"{RED}Selected model not available.{RESET}"
        
        if self.current_model == "t5":
            result = model.generate_text(text, device=str(self.device), **self.settings)
        else:
            # Simplified custom model generation
            result = f"[Custom model response to: {text[:50]}...]"
        
        return result if result and result.strip() else "(model returned empty response)"

    def _translate_sanskrit(self, text: str) -> str:
        if self.translator is None:
            return f"{RED}Translation model not available.{RESET}"
        
        try:
            tok = self.translator["tokenizer"]
            mdl = self.translator["model"]
            
            tok.src_lang = "eng_Latn"
            inputs = tok(text, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                generated = mdl.generate(
                    **inputs,
                    forced_bos_token_id=tok.convert_tokens_to_ids("san_Deva"),
                    num_beams=5,
                    max_length=256,
                )
            result = tok.batch_decode(generated, skip_special_tokens=True)[0]
            
            # Add transliteration
            transliterated = self._transliterate_sanskrit(result)
            return f"{GREEN}Sanskrit:{RESET} {result}\\n{BLUE}Transliteration:{RESET} {transliterated}"
        except Exception as e:
            return f"{RED}Translation error: {e}{RESET}"

    def _transliterate_sanskrit(self, devanagari_text):
        """Basic Devanagari to Latin transliteration."""
        mapping = {
            'अ': 'a', 'आ': 'ā', 'इ': 'i', 'ई': 'ī', 'उ': 'u', 'ऊ': 'ū',
            'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
            'क': 'ka', 'ख': 'kha', 'ग': 'ga', 'घ': 'gha', 'ङ': 'ṅa',
            'च': 'ca', 'छ': 'cha', 'ज': 'ja', 'झ': 'jha', 'ञ': 'ña',
            'त': 'ta', 'थ': 'tha', 'द': 'da', 'ध': 'dha', 'न': 'na',
            'प': 'pa', 'फ': 'pha', 'ब': 'ba', 'भ': 'bha', 'म': 'ma',
            'य': 'ya', 'र': 'ra', 'ल': 'la', 'व': 'va', 'श': 'śa', 
            'ष': 'ṣa', 'स': 'sa', 'ह': 'ha', 'ं': 'ṃ', 'ः': 'ḥ',
            '्': '', ' ': ' ', '।': '.'
        }
        return "".join(mapping.get(char, char) for char in devanagari_text)

    def _summarize(self, text: str) -> str:
        prompt = f"summarize: {text}"
        result = self.t5_model.generate_text(prompt, device=str(self.device), **self.settings)
        return result if result and result.strip() else "(summarization failed)"

    def _question_answering(self, text: str) -> str:
        if "context:" not in text.lower():
            return f"{YELLOW}Please provide format: Question: [your question] Context: [context text]{RESET}"
        
        # Format for T5: "question: <question> context: <context>"
        formatted = text.lower().replace("q:", "question:").replace("a:", "answer:")
        result = self.t5_model.generate_text(formatted, device=str(self.device), **self.settings)
        return result if result and result.strip() else "(no answer found)"

    def _analyze_sentiment(self, text: str) -> str:
        if self.sentiment_pipeline is None:
            return f"{RED}Sentiment model not available.{RESET}"
        
        try:
            result = self.sentiment_pipeline(text)[0]
            label = result['label']
            confidence = result['score']
            
            emoji_map = {"POSITIVE": "😊", "NEGATIVE": "😞", "NEUTRAL": "😐"}
            emoji = emoji_map.get(label, "🤔")
            
            if confidence > 0.8:
                conf_desc = "very confident"
            elif confidence > 0.6:
                conf_desc = "confident"
            else:
                conf_desc = "uncertain"
            
            return f"{emoji} {BOLD}{label}{RESET} ({conf_desc}: {confidence:.3f})"
        except Exception as e:
            return f"{RED}Sentiment error: {e}{RESET}"

    def _extract_entities(self, text: str) -> str:
        if self.ner_pipeline is None:
            return f"{RED}NER model not available.{RESET}"
        
        try:
            entities = self.ner_pipeline(text)
            if not entities:
                return f"{YELLOW}No named entities found.{RESET}"
            
            result = []
            entity_types = {"PER": "👤", "ORG": "🏢", "LOC": "📍", "MISC": "🏷️"}
            
            for ent in entities:
                emoji = entity_types.get(ent['entity_group'], "🔖")
                result.append(
                    f"{emoji} {BOLD}{ent['word']}{RESET} → {ent['entity_group']} "
                    f"({ent['score']:.2f})"
                )
            return "\\n".join(result)
        except Exception as e:
            return f"{RED}NER error: {e}{RESET}"

    def _classify(self, text: str) -> str:
        try:
            enc = self.t5_model.encode_text(text, device=str(self.device))
            with torch.no_grad():
                logits = self.t5_model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    task="classification",
                )
            probs = F.softmax(logits, dim=-1)[0]
            top_class = probs.argmax().item()
            confidence = probs[top_class].item()
            top5 = torch.topk(probs, min(5, len(probs)))
            
            lines = [f"{BOLD}Predicted class:{RESET} {top_class} ({confidence:.1%} confidence)\\n"]
            for i, (cls, p) in enumerate(zip(top5.indices.tolist(), top5.values.tolist())):
                bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
                lines.append(f"  [{cls:>2}] {bar} {p:.1%}")
            
            # Reset to generation after classification
            self.task = "generation"
            lines.append(f"\\n{CYAN}(Switched back to generation mode){RESET}")
            
            return "\\n".join(lines)
        except Exception as e:
            self.task = "generation"
            return f"{RED}Classification error: {e}{RESET}"

    # ── Command handlers ─────────────────────────────────────────
    def handle_command(self, cmd: str) -> tuple[bool, str]:
        cmd = cmd.strip().lower()
        
        if cmd == "/exit":
            print(f"\\n{CYAN}Thanks for using JnanaVerse! — Dharmin Joshi / DevKay{RESET}\\n")
            sys.exit(0)
        
        elif cmd == "/help":
            return True, BANNER
        
        elif cmd in ["/generate", "/gen"]:
            self.task = "generation"
            return True, f"{GREEN}Switched to: {TASK_LABELS['generation']}{RESET}"
        
        elif cmd in ["/translate", "/sanskrit", "/san"]:
            self.task = "translate"
            return True, f"{GREEN}Switched to: {TASK_LABELS['translate']}{RESET}\\nJust type English text to translate."
        
        elif cmd in ["/summarize", "/summary", "/sum"]:
            self.task = "summarize"
            return True, f"{GREEN}Switched to: {TASK_LABELS['summarize']}{RESET}\\nPaste text to summarize."
        
        elif cmd in ["/qa", "/question"]:
            self.task = "qa"
            return True, f"{GREEN}Switched to: {TASK_LABELS['qa']}{RESET}\\nFormat: Question: [your question] Context: [context]"
        
        elif cmd in ["/sentiment", "/emotion"]:
            self.task = "sentiment"
            return True, f"{GREEN}Switched to: {TASK_LABELS['sentiment']}{RESET}\\nEnter text to analyze sentiment."
        
        elif cmd in ["/ner", "/entities"]:
            self.task = "ner"
            return True, f"{GREEN}Switched to: {TASK_LABELS['ner']}{RESET}\\nEnter text to extract named entities."
        
        elif cmd in ["/classify", "/classification"]:
            self.task = "classify"
            return True, f"{YELLOW}Switched to: {TASK_LABELS['classify']}{RESET}\\nEnter text to classify (will auto-return to generation)."
        
        elif cmd in ["/similarity", "/sim"]:
            self.task = "similarity"
            return True, f"{GREEN}Switched to: {TASK_LABELS['similarity']}{RESET}\\nEnter first sentence:"
        
        elif cmd == "/model":
            if self.custom_model is None:
                return True, f"{YELLOW}Only T5 model available.{RESET}"
            
            self.current_model = "custom" if self.current_model == "t5" else "t5"
            model_name = "Custom Transformer" if self.current_model == "custom" else "T5"
            return True, f"{GREEN}Switched to: {model_name}{RESET}"
        
        elif cmd == "/settings":
            settings_str = "\\n".join([
                f"{BOLD}Current Settings:{RESET}",
                f"  Temperature: {self.settings['temperature']}",
                f"  Max tokens: {self.settings['max_new_tokens']}",
                f"  Beams: {self.settings['num_beams']}",
                f"  Top-k: {self.settings['top_k']}",
                f"  Top-p: {self.settings['top_p']}",
                "",
                f"{DIM}Use /set <parameter> <value> to change{RESET}"
            ])
            return True, settings_str
        
        elif cmd.startswith("/set "):
            return self._handle_setting_change(cmd)
        
        elif cmd == "/history":
            return True, self._format_history()
        
        elif cmd == "/export":
            return True, self._export_conversation()
        
        elif cmd == "/clear":
            self.conversation_history = []
            return True, f"{GREEN}Conversation history cleared.{RESET}"
        
        elif cmd == "/task":
            model_info = f" ({self.current_model.upper()})" if self.current_model else ""
            return True, f"Current: {BOLD}{TASK_LABELS.get(self.task, self.task)}{model_info}{RESET}"
        
        elif cmd == "/info":
            info = [
                f"{BOLD}System Information:{RESET}",
                f"  Device: {self.device}",
                f"  Current task: {self.task}",
                f"  Model: {self.current_model}",
                f"  T5 params: {count_parameters(self.t5_model) if self.t5_model else 'N/A'}",
                f"  Custom params: {count_parameters(self.custom_model) if self.custom_model else 'N/A'}",
                f"  Conversation length: {len(self.conversation_history)} messages"
            ]
            return True, "\\n".join(info)
        
        return False, f"{RED}Unknown command '{cmd}'. Type /help for options.{RESET}"

    def _handle_setting_change(self, cmd: str) -> tuple[bool, str]:
        parts = cmd.split()
        if len(parts) != 3:
            return True, f"{RED}Usage: /set <parameter> <value>{RESET}"
        
        param, value = parts[1], parts[2]
        
        try:
            if param == "temperature":
                self.settings["temperature"] = max(0.1, min(2.0, float(value)))
            elif param in ["max_tokens", "max_new_tokens"]:
                self.settings["max_new_tokens"] = max(16, min(512, int(value)))
            elif param in ["beams", "num_beams"]:
                self.settings["num_beams"] = max(1, min(10, int(value)))
            elif param in ["topk", "top_k"]:
                self.settings["top_k"] = max(1, min(100, int(value)))
            elif param in ["topp", "top_p"]:
                self.settings["top_p"] = max(0.1, min(1.0, float(value)))
            else:
                return True, f"{RED}Unknown parameter '{param}'. Available: temperature, max_tokens, beams, top_k, top_p{RESET}"
            
            return True, f"{GREEN}Set {param} = {self.settings.get(param, value)}{RESET}"
        except ValueError:
            return True, f"{RED}Invalid value for {param}: {value}{RESET}"

    def _format_history(self) -> str:
        if not self.conversation_history:
            return f"{YELLOW}No conversation history.{RESET}"
        
        lines = [f"{BOLD}Conversation History ({len(self.conversation_history)} messages):{RESET}\\n"]
        for i, msg in enumerate(self.conversation_history[-10:], 1):  # Show last 10
            timestamp = msg["timestamp"].strftime("%H:%M:%S")
            role = msg["role"]
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            
            if role == "user":
                lines.append(f"{DIM}[{timestamp}]{RESET} {BLUE}You:{RESET} {content}")
            else:
                lines.append(f"{DIM}[{timestamp}]{RESET} {GREEN}Bot:{RESET} {content}")
        
        if len(self.conversation_history) > 10:
            lines.append(f"\\n{DIM}... showing last 10 messages{RESET}")
        
        return "\\n".join(lines)

    def _export_conversation(self) -> str:
        if not self.conversation_history:
            return f"{YELLOW}No conversation to export.{RESET}"
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"jnanaverse_chat_{timestamp}.txt"
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("JnanaVerse Conversation Export\\n")
                f.write(f"Generated: {datetime.datetime.now()}\\n")
                f.write("=" * 50 + "\\n\\n")
                
                for msg in self.conversation_history:
                    timestamp = msg["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                    role = msg["role"].upper()
                    task = msg.get("task", "unknown")
                    model = msg.get("model", "unknown")
                    content = msg["content"]
                    
                    f.write(f"[{timestamp}] {role} ({task}/{model})\\n")
                    f.write(f"{content}\\n\\n")
            
            return f"{GREEN}Conversation exported to: {filename}{RESET}"
        except Exception as e:
            return f"{RED}Export failed: {e}{RESET}"

    # ── Similarity task with two-step flow ──────────────────────
    def handle_similarity_task(self, user_input: str) -> str:
        if not hasattr(self, "sim_step"):
            self.sim_step = 0
        
        if self.sim_step == 0:
            self.sim_sent1 = user_input
            self.sim_step = 1
            return f"{GREEN}Sentence 1 stored. Now enter sentence 2:{RESET}"
        else:
            sim_sent2 = user_input
            result = self._compute_similarity(self.sim_sent1, sim_sent2)
            self.sim_step = 0
            return result + f"\\n\\n{GREEN}Enter sentence 1 for next comparison:{RESET}"

    def _compute_similarity(self, s1: str, s2: str) -> str:
        try:
            e1 = self.t5_model.encode_text(s1, device=str(self.device))
            e2 = self.t5_model.encode_text(s2, device=str(self.device))
            
            with torch.no_grad():
                score = self.t5_model(
                    input_ids=(e1["input_ids"], e2["input_ids"]),
                    attention_mask=(e1["attention_mask"], e2["attention_mask"]),
                    task="similarity",
                ).item()
            
            bar_len = int((score + 1) / 2 * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            
            if score > 0.85:
                verdict = f"{GREEN}very similar{RESET}"
            elif score > 0.5:
                verdict = f"{YELLOW}somewhat similar{RESET}"
            elif score > 0.2:
                verdict = f"{BLUE}loosely related{RESET}"
            else:
                verdict = f"{RED}dissimilar{RESET}"
            
            return (
                f'{BOLD}Sentence 1:{RESET} "{s1}"\\n'
                f'{BOLD}Sentence 2:{RESET} "{s2}"\\n\\n'
                f"Cosine similarity: {score:.4f}\\n"
                f"[{bar}]\\n"
                f"Verdict: {verdict}"
            )
        except Exception as e:
            return f"{RED}Similarity computation error: {e}{RESET}"

    # ── Main conversation loop ───────────────────────────────────
    def run(self):
        print(BANNER)
        
        while True:
            try:
                task_label = TASK_LABELS.get(self.task, self.task)
                model_indicator = f" ({self.current_model.upper()})" if self.current_model else ""
                prompt = f"{BOLD}[{task_label}{model_indicator}]{RESET} {BLUE}You{RESET} › "
                user_input = input(prompt).strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\\n{CYAN}Goodbye! — Dharmin Joshi / DevKay{RESET}\\n")
                break
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                handled, msg = self.handle_command(user_input)
                if msg:
                    print(f"\\n{msg}\\n")
                continue
            
            # Handle similarity task special flow
            if self.task == "similarity":
                response = self.handle_similarity_task(user_input)
                print(f"\\n{GREEN}JnanaVerse{RESET} › {response}\\n")
                continue
            
            # Regular task handling
            print(f"\\n{GREEN}JnanaVerse{RESET} › ", end="", flush=True)
            response = self.respond(user_input)
            print(f"{response}\\n")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="JnanaVerse Enhanced CLI Chatbot")
    p.add_argument("--t5-model", default="t5-small", help="T5 model name")
    p.add_argument("--custom-vocab", default=5000, type=int, help="Custom transformer vocab size")
    p.add_argument("--classes", default=5, type=int, help="Number of classification classes")
    p.add_argument("--device", default="auto", help="Device: auto | cpu | cuda | mps")
    p.add_argument("--seed", default=42, type=int, help="Random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    
    bot = EnhancedJnanaChat(
        t5_model=args.t5_model,
        custom_vocab=args.custom_vocab,
        num_classes=args.classes,
        device_str=args.device,
    )
    bot.run()
