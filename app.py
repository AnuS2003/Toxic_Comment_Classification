import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pytesseract
from PIL import Image
import re

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the saved model and tokenizer
model_name = "Saved_model"
Bert_Tokenizer = BertTokenizer.from_pretrained(model_name)
Bert_Model = BertForSequenceClassification.from_pretrained(model_name)

# Function to predict toxicity
def predict_toxicity(text):
    inputs = Bert_Tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        outputs = Bert_Model(input_ids, attention_mask=attention_mask)
        
    logits = outputs.logits
    prediction = torch.sigmoid(logits).cpu().numpy()[0]
    
    result = {
        'toxic': int(prediction[0] >= 0.5),
        'severe_toxic': int(prediction[1] >= 0.5),
        'obscene': int(prediction[2] >= 0.5),
        'threat': int(prediction[3] >= 0.5),
        'insult': int(prediction[4] >= 0.5),
        'identity_hate': int(prediction[5] >= 0.5)
    }
    
    return result, prediction

# Function to extract toxic keywords
def extract_toxic_keywords(text):
    keywords = {
        'toxic': ['bad', 'awful', 'horrible', 'FUCK', 'NIGGER NIGGER', 'hate', 'know', 'article', 'people','page'],
        'severe_toxic': ['disgusting', 'vile', 'FUCK', 'SHIT SHIT', 'ass', 'Go fuck', 'DIE DIE', 'faggot faggot', 'fuck yourselfgo', 'yourselfgo fuck', 'MUST DIE'],
        'obscene': ['FUCK', 'NIGGER NIGGER', 'fucking', 'know', 'SHIT SHIT', 'ass', 'DICKS DICK', 'Go fuck', 'youfuck youfuck', 'page'],
        'threat': ['hurt', 'ass', 'DIE DIE', 'MUST DIE', 'KILL', 'going', 'TRAITOR VANDAL', 'VANDAL MUST', 'DIE FUK', 'JIM WALE'],
        'insult': ['idiot', 'stupid', 'fuck', 'NIGGER NIGGER', 'fucking', 'know', 'moron hi', 'hi moron', 'JEW FAT', 'FAT JEW', 'HATE', 'Go fuck'],
        'identity_hate': ['racist', 'bigot', 'NIGGER NIGGER', 'JEW FAT', 'FAT JEW', 'DIE DIE', 'fucking', 'NIGGERJEW NIGGERJEW', 'HUGE FAGGOT', 'FAGGOT HUGE', 'Bitch Ass', 'Please Nigga']
    }
    toxic_parts = {}
    for key, word_list in keywords.items():
        toxic_parts[key] = [word for word in word_list if re.search(rf'\b{re.escape(word)}\b', text, re.IGNORECASE)]
    return toxic_parts

# Function to provide feedback based on toxic parts
def provide_feedback(toxic_parts):
    feedback_messages = {
        'toxic': 'Your comment contains toxic language. Please use more respectful and constructive words.',
        'severe_toxic': 'This comment is extremely harmful. Avoid using severe and aggressive language.',
        'obscene': 'Your comment includes obscene content. Refrain from using inappropriate or vulgar terms.',
        'threat': 'Threatening language is unacceptable. Ensure that your comments do not include any threats or intimidation.',
        'insult': 'Insulting remarks are harmful. Always strive to communicate with respect and avoid personal attacks.',
        'identity_hate': 'Comments with identity hate are discriminatory and unacceptable. Avoid language that targets individuals based on their identity.'
    }
    
    feedback = ""
    for category, words in toxic_parts.items():
        if words:
            feedback += f"<p style='font-size:14px;'><strong>{category.replace('_', ' ').capitalize()}:</strong> {', '.join(words)}. {feedback_messages[category]}</p>"
    return feedback

# Function to extract text from image
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text

# Streamlit UI
st.title("Toxic Comment Classification")

# Text input
user_input = st.text_area("Enter your comment here:")

# Image upload
uploaded_image = st.file_uploader("Or upload an image containing the comment:", type=["jpg", "jpeg", "png"])

if st.button("Predict"):
    if user_input or uploaded_image:
        with st.spinner("Analyzing..."):
            if user_input:
                text_to_analyze = user_input
            else:
                image = Image.open(uploaded_image)
                text_to_analyze = extract_text_from_image(image)
                st.write("Extracted Text:")
                st.write(text_to_analyze)
            
            if text_to_analyze:
                prediction, _ = predict_toxicity(text_to_analyze)
                toxic_parts = extract_toxic_keywords(text_to_analyze)
                feedback = provide_feedback(toxic_parts)
                
                if prediction['toxic']:
                    result_message = "<p style='font-size:20px; font-weight:bold; color:red;'>The comment is toxic.</p>"
                    toxic_categories = [category.replace('_', ' ') for category in ["severe_toxic", "obscene", "threat", "insult", "identity_hate"] if prediction[category]]
                    
                    if toxic_categories:
                        result_message += "<p style='font-size:18px;'>The comment belongs to the following categories: <span style='color:blue;'>" + ", ".join(toxic_categories) + "</span>.</p>"
                        
                        if feedback:
                            result_message += f"<p style='font-size:16px;'>Key toxic words identified:</p>{feedback}"
                    else:
                        result_message += "<p style='font-size:18px;'>No specific toxic words were identified for the categories.</p>"
                    
                else:
                    result_message = "<p style='font-size:20px; font-weight:bold; color:green;'>The comment is non-toxic.</p>"
                
                st.markdown(result_message, unsafe_allow_html=True)
            else:
                st.error("No text found in the image.")
    else:
        st.error("Please enter a comment or upload an image to classify.")
