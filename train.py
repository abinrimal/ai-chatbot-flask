from datasets import Dataset
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Tokenizer

texts = [
    {"text": "Hello, how are you?"},
    {"text": "I am training a GPT-2 model."},
    {"text": "This is a simple example."},
    {"text": "You can add as many sentences as you want."},
    {"text": "Training language models is fun!"},
]

raw_datasets = Dataset.from_list(texts).train_test_split(test_size=0.2)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Add this line to fix padding error:
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=32,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text"])

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    logging_steps=10,
    save_strategy="epoch",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)
