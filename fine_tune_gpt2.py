import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

MODEL_NAME = "gpt2"
FILE_PATH = "data/train_data.jsonl"

def main():
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))

    # Load dataset
    dataset = load_dataset("json", data_files={"train": FILE_PATH}, split="train")
    train_val = dataset.train_test_split(test_size=0.1)
    print(f"Loaded dataset with {len(dataset)} samples.")

    # Tokenize
    def tokenize(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=1024,
            padding="max_length"
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = train_val.map(tokenize, batched=True, remove_columns=["text"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        logging_dir="./logs",
        report_to="tensorboard",
        evaluation_strategy="epoch"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"]
    )

    # Train and save
    trainer.train()
    trainer.save_model("./fine_tuned_gpt2")
    tokenizer.save_pretrained("./fine_tuned_gpt2")
    print("✅ Fine-tuning complete. Model saved to ./fine_tuned_gpt2")

if __name__ == "__main__":
    main()
