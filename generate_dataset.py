import json
import random

STOP_TOKEN = ""  # GPT-2's stop token

# Example conversation templates (expand this list)
conversations = [
    [
        ("User: Hello!", "AI: Hi there! How can I assist you today?"),
        ("User: Can you tell me a joke?", "AI: Why don't scientists trust atoms? Because they make up everything!"),
        ("User: Haha, good one.", "AI: Glad you liked it!")
    ],
    [
        ("User: What's your name?", "AI: I'm your AI assistant. How can I help you today?"),
        ("User: Can you help me with my homework?", "AI: Sure! What subject do you need help with?"),
        ("User: Math.", "AI: Great! What math problem are you working on?")
    ],
    [
        ("User: Tell me something interesting.", "AI: Did you know honey never spoils? Archaeologists found 3000-year-old honey in Egyptian tombs that was still edible!"),
        ("User: That's amazing.", "AI: Yes, nature can be pretty fascinating.")
    ]
]

def generate_jsonl(filename, num_samples=100):
    with open(filename, "w") as f:
        for _ in range(num_samples):
            convo = random.choice(conversations)
            convo_text = ""
            for user, ai in convo:
                # Append stop token to AI response
                ai_with_stop = f"{ai} {STOP_TOKEN}"
                convo_text += f"{user}\n{ai_with_stop}\n"
            # Remove trailing newline and save as one text sample
            convo_text = convo_text.strip()
            json_line = json.dumps({"text": convo_text})
            f.write(json_line + "\n")

if __name__ == "__main__":
    generate_jsonl("data/train_data.jsonl", num_samples=1000)
    print("Generated 1000 synthetic conversation samples in data/train_data.jsonl")
