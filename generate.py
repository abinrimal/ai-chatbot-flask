import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

conversation = ""

print("Start chatting with the AI! Type 'exit' or 'quit' to stop.\n")

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Prompt with instruction
    conversation = "The following is a conversation between a helpful AI assistant and a human.\n"
    conversation += f"User: {user_input}\nAI:"

    inputs = tokenizer(conversation, return_tensors="pt", truncation=True, max_length=512)
    input_length = inputs["input_ids"].shape[1]

    output = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=input_length + 200,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.85,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_tokens = output[0][input_length:]
    ai_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    print(f"AI: {ai_response}")
