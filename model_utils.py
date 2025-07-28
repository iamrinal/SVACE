import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
from joblib import load
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib
import os
from retrieval_utils import query_index

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load classifier
def load_classifier(model_dir="saved_classifier"):
    tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model

# Load generator
def load_generator(model_dir="saved_generator"):
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model

# Load label encoder
def load_encoder(encoder_path="label_encoder.joblib"):
    return load(encoder_path)

# Classification
def classify(code_snippet, flag, tokenizer, model, encoder):
    input_text = f"[{flag}] {code_snippet}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_label_idx = torch.argmax(probs, dim=1).item()
    pred_label = encoder.inverse_transform([pred_label_idx])[0]
    return pred_label, probs[0][pred_label_idx].item()

# Retrieval
retrieval_model = None
def retrieve_similar(code_snippet, flag, top_k=3):
    input_text = f"[{flag}] {code_snippet}"
    return query_index(input_text, top_k=top_k)

# Generation
def generate_fix(code_snippet, flag, generator_tokenizer, generator_model, retrieved_examples=None):
    input_text = f"fix: [{flag}] {code_snippet}"
    if retrieved_examples:
        for ex in retrieved_examples:
            input_text += f"\n# Similar: [{ex['flag']}] {ex['code_with_bug']}\n# Fix: {ex['fixed_code']}"
    inputs = generator_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = generator_model.generate(**inputs, max_length=256, num_beams=4, early_stopping=True)
    fix = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return fix

# Full pipeline
def classify_and_fix(code_snippet, flag, classifier_tokenizer, classifier_model, encoder, generator_tokenizer, generator_model, top_k=3):
    label, confidence = classify(code_snippet, flag, classifier_tokenizer, classifier_model, encoder)
    retrieved = retrieve_similar(code_snippet, flag, top_k=top_k)
    fix = generate_fix(code_snippet, flag, generator_tokenizer, generator_model, retrieved)
    return label, confidence, fix, retrieved
