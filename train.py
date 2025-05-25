import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

MODEL_NAME = "gpt2"
FILE_PATH = "data/train_data.jsonl"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_datasets = load_dataset("json", data_files={"train": FILE_PATH}, split="train")
    train_val = raw_datasets.train_test_split(test_size=0.1)

    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
            padding="max_length"
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    tokenized_datasets = train_val.map(tokenize_function, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
        logging_dir="./logs",
        report_to="tensorboard",
        #evaluation_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer
    )

    trainer.train(resume_from_checkpoint=True)
    trainer.save_model("./fine_tuned_gpt2")
    tokenizer.save_pretrained("./fine_tuned_gpt2")

if __name__ == "__main__":
    main()
