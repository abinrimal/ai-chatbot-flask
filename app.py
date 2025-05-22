from flask import Flask, render_template, request, session, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
import json

app = Flask(__name__)
app.secret_key = 'replace_this_with_a_random_secret_key'

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Load your fine-tuned model and tokenizer (same as generate.py)
tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")
model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2").to(device)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

MEMORY_FILE = "chat_memory.json"

def load_history():
    if 'chat_history' in session:
        return session['chat_history']
    elif os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            chat_history = json.load(f).get("history", "")
        session['chat_history'] = chat_history
        return chat_history
    else:
        return ""

def save_history(history):
    session['chat_history'] = history
    with open(MEMORY_FILE, "w") as f:
        json.dump({"history": history}, f, indent=2)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"response": "Please enter a message."})

    if user_input.lower() in ["exit", "quit"]:
        save_history("")
        return jsonify({"response": "Goodbye! Chat history cleared."})

    # Follow your generate.py style prompt
    prompt = "The following is a conversation between a helpful AI assistant and a human.\n"
    prompt += f"User: {user_input}\nAI:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    input_length = input_ids.shape[1]

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=input_length + 200,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.5,   # lower temperature for less randomness
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
    )

    generated_tokens = output_ids[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Save current prompt + response as new history (optional)
    chat_history = f"{prompt} {response}\n"
    save_history(chat_history)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
