# AI-Powered-Healthcare-Chatbot
## Overview

This project implements a Healthcare Assistant Chatbot using a fine-tuned distilGPT2 model. It provides a simple interface for users to ask healthcare-related questions and receive responses. The chatbot combines keyword-based responses for common queries with a fine-tuned language model for more complex questions. It's built with Streamlit for a user-friendly web interface.

## Key Features

* **Fine-tuned distilGPT2:** Leverages a pre-trained language model, distilGPT2, further trained on a healthcare-related dataset (`shaneperry0101/health-chatbot` from Hugging Face) for more relevant and accurate responses.
* **Keyword-based responses:** Handles common queries related to symptoms, appointments, and medication using predefined responses for quick and efficient answers.
* **Streamlit UI:** Provides an interactive web interface for users to easily interact with the chatbot.
* **Error Handling:** Includes basic error handling to gracefully manage potential issues during response generation.
* **Customizable Parameters:** Allows adjustment of model parameters like `max_length`, `temperature`, `top_k`, `top_p`, and `repetition_penalty` to fine-tune the chatbot's behavior.
* **Model Training Included:** The `fine_tuned_model.py` script handles the training of the distilGPT2 model, so you can easily recreate the fine-tuned model.  It also checks for the existing model to avoid retraining.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Soumya-Choudhury/AI-Powered-Healthcare-Chatbot
   cd healthcare-chatbot

2. **Setup virtual environment:**
   ```bash
   python3 -m venv .venv  # Or use virtualenv or conda
   source .venv/bin/activate  # On Linux/macOS
   .\venv\Scripts\activate  # On Windows

3. **Install all the dependencies:**
   ```bash
   pip install -r requirements.txt

4. **Run the fine_tuned_model.py file first:**
   ```bash
   python fine_tuned_model.py

5. **Run the app.py file:**
   ```bash
   streamlit run app.py   #It will launch the appon web browser

