#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import re

# Display a robot image at the top of your app
robot_image = 'chatbot_image.webp'  # Update this path to your downloaded robot image file
st.image(robot_image, caption="Chatbot ready to assist you")

# Use Markdown for colorful headings and text
st.markdown("<h1 style='color: blue;'>Mental Health Support Chatbot</h1>", unsafe_allow_html=True)

# Load your model and tokenizer
def load_model_and_tokenizer():
    model_dir = "./model"  # Ensure this points to the directory where your model is saved
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    chatbot = pipeline('text-generation', model=model, tokenizer=tokenizer)
    return chatbot

# Initialize Sentiment Analysis Pipeline
sentiment_model = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Function to Trim Generated Text to Last Complete Sentence
def trim_to_last_sentence(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    if sentences and not text.endswith(('.', '?', '!')):
        sentences = sentences[:-1]
    return ' '.join(sentences)

def main():
    chatbot = load_model_and_tokenizer()

    with st.form("chat_form"):
        user_input = st.text_input("Talk to the chatbot:", "")
        submit_button = st.form_submit_button(label='Send')
        end_chat = st.form_submit_button(label='End Chat')

    if submit_button and user_input:
        sentiment_result = sentiment_model(user_input)
        sentiment = sentiment_result[0]['label']
        
        if sentiment == 'NEGATIVE':
            st.error("It sounds like you're going through a tough time. Let's see if I can help.")
        else:
            st.success("That's good to hear! How can I assist you today?")
        
        generated_responses = chatbot(user_input, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
        response = generated_responses[0]['generated_text']
        response_trimmed = trim_to_last_sentence(response)
        st.write(response_trimmed)

    if end_chat:
        st.write("Thank you for chatting. Take care!")

    # Feedback collection
    feedback = st.text_area("Feedback on the advice (Optional):")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")

if __name__ == "__main__":
    main()
