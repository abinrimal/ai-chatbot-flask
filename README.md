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
├── results/                  # Model checkpoints (usually excluded from git)
├── fine_tuned_gpt2/          # Fine-tuned model artifacts
├── chat_sessions/            # Saved chat session JSON files
├── templates/                # Flask HTML templates
│   └── index.html
├── app.py                   # Flask app for chatting
├── fine_tune_gpt2.py         # Script to fine-tune GPT-2
├── generate_dataset.py       # Script to generate training datasets
├── generate.py               # Script for generating text from the model
├── train.py                  # Training utility script
├── requirements.txt          # Python dependencies
└── README.txt                # This file

## Setup Instructions

# Prerequisites

- Python 3.8+
- Git
- pip package manager

# Install dependencies

pip install -r requirements.txt

# Running the Chatbot Locally

Start the Flask app:

python app.py

# Access the chat interface:

Open your browser at http://127.0.0.1:PORT

Chat with the fine-tuned GPT-2 model

You can enter messages and receive AI-generated responses.

## Fine-tuning the Model

Prepare your training dataset or generate it using:

python generate_dataset.py

Start training with:

python fine_tune_gpt2.py

Checkpoints will be saved in the results/ directory.

# Generating Text with the Model

Use:

python generate.py --prompt "Your prompt here"

to get model completions.

## License

This project is open source and free to use under the MIT License.

Created by Abin Rimal.
