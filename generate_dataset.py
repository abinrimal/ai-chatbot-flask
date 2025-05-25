import json
import random
import os

# Example conversation templates
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
        ("User: Tell me something interesting.", "AI: Did you know honey never spoils?"),
        ("User: That's amazing.", "AI: Yes, nature can be pretty fascinating.")
    ]
]

def generate_jsonl(filename="data/train_data.jsonl", num_samples=1000):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        for _ in range(num_samples):
            convo = random.choice(conversations)
            convo_text = ""
            for user, ai in convo:
                convo_text += f"{user}\n{ai}\n"
            json_line = json.dumps({"text": convo_text.strip()})
            f.write(json_line + "\n")

if __name__ == "__main__":
    generate_jsonl()
    print("Generated dataset: data/train_data.jsonl")
