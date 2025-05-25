### AI Chat Model Project

## Overview

This project is an AI chatbot built using GPT-2 fine-tuning. It includes training scripts, dataset generation, and a Flask-based web app for interaction. The project structure supports training, inference, and managing chat sessions.

## What It Does

- Fine-tunes a GPT-2 model with your custom dataset.
- Saves checkpoints during training.
- Provides a Flask web app (app.py) to chat with the model.
- Stores chat sessions in JSON files.
- Supports generating new datasets and training from scratch.
- Includes templates and scripts for model interaction.

## Project Structure

ai_chat_model/
├── results/ # Model checkpoints (usually excluded from git)
├── fine_tuned_gpt2/ # Fine-tuned model artifacts
├── chat_sessions/ # Saved chat session JSON files
├── templates/ # Flask HTML templates
│ └── index.html
├── app.py # Flask app for chatting
├── fine_tune_gpt2.py # Script to fine-tune GPT-2
├── generate_dataset.py # Script to generate training datasets
├── generate.py # Script for generating text from the model
├── train.py # Training utility script
├── requirements.txt # Python dependencies
└── README.md # This file

## Setup Instructions

# Prerequisites

- Python 3.8+
- Git
- pip package manager

# Install dependencies
```bash
pip install -r requirements.txt
```

# Running the Chatbot Locally

Start the Flask app:
```bash
python3 app.py
```
# Access the chat interface:

Open your browser at http://127.0.0.1:PORT

Chat with the fine-tuned GPT-2 model

You can enter messages and receive AI-generated responses.

## Fine-tuning the Model

1) Prepare your training dataset or generate it using:
```bash
python generate_dataset.py
```
This will create a data/train_data.jsonl file with your training examples.


2) Train The Model:

```bash
python train.py
```
The script fine-tunes GPT-2 on your dataset.

Model checkpoints and final artifacts are saved in results/ and fine_tuned_gpt2/.

Logs are saved to ./logs and can be monitored with TensorBoard.

Training Duration
Training time varies based on:
-> Dataset size (more data = longer training)
-> Model size (GPT-2 base is faster than larger versions)
-> Hardware (GPU > MPS (Apple Silicon) > CPU)
-> Training parameters (batch size, number of epochs)

Typical times:
-> GPU (e.g., RTX 3060): ~10-30 minutes for 1000 samples, 5 epochs
-> Apple Silicon MPS: ~2-4x slower than GPU
-> CPU: Much slower, potentially hours or more

# Optional: Monitor Training with TensorBoard

1) Install TensorBoard (if not installed):

```bash
pip3 install tensorboard
```

2) Run TensorBoard:

```bash
tensorboard --logdir=./logs
```

Open the URL shown in the terminal (usually http://localhost:6006) in your browser to visualize training metrics.

## Generating Text with the Model
Generate completions from your fine-tuned model using:

```bash
python generate.py --prompt "Your prompt here"
```

## License

This project is open source and free to use under the MIT License.

Created by Abin Rimal.
