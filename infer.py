import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_DIR = "./fine_tuned_gpt2"

def load_model_and_tokenizer(model_dir):
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

def chat_loop(tokenizer, model, max_length=150):
    history = ""

    print("🗨️  Chat with your fine-tuned model (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        history += f"User: {user_input}\nAI:"

        inputs = tokenizer.encode(history, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_length,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the latest AI response
        ai_reply = response[len(history):].split("User:")[0].strip()
        print(f"AI: {ai_reply}")
        history += f" {ai_reply}\n"

if __name__ == "__main__":
    tokenizer, model = load_model_and_tokenizer(MODEL_DIR)
    chat_loop(tokenizer, model)
