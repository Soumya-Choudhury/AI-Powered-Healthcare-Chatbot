# distilgpt2 retrained
import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Load fine-tuned model and tokenizer
MODEL_PATH = "./fine_tuned_distilgpt2"

@st.cache_resource()
def load_chatbot():
    if os.path.exists(MODEL_PATH):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    else:
        st.error("Fine-tuned model not found! Make sure to train and save the model first.")
        return None

chatbot = load_chatbot()

# Define Healthcare Chatbot Logic
def healthcare_chatbot(user_input):
    user_input = user_input.lower()
    
    keyword_responses = {
        "symptom": "It seems like you're experiencing symptoms. Please consult a doctor.",
        "appointment": "Would you like me to schedule an appointment?",
        "medication": "Take your prescribed medications regularly. If you have concerns, consult your doctor."
    }
    
    # Check for keyword-based responses first
    for keyword, response in keyword_responses.items():
        if keyword in user_input:
            return response
    
    # Generate model-based response
    if chatbot:
        try:
            response = chatbot(
    user_input, 
    max_length=150,  
    num_return_sequences=1, 
    temperature=0.7,  
    top_k=50,         
    top_p=0.9,        
    repetition_penalty=1.2  
)

            return response[0]['generated_text'].strip()
        except Exception as e:
            return f"Error generating response: {e}"
    else:
        return "Chatbot is unavailable at the moment."

# Streamlit UI
def main():
    st.title("🩺 Healthcare Assistant Chatbot")
    st.markdown("👋 **Ask me any healthcare-related question!**")
    
    user_input = st.text_input("How can I assist you today?", "")
    
    if st.button("Submit"):
        if user_input.strip():
            response = healthcare_chatbot(user_input)
            st.write(f"**Healthcare Assistant:** {response}")
        else:
            st.warning("⚠️ Please enter a query.")

if __name__ == "__main__":
    main()
