from flask import Flask, render_template, request
import joblib
import os
from model_utils import load_classifier, load_generator, load_encoder, classify_and_fix

app = Flask(__name__)

# Load models and encoder
classifier_tokenizer, classifier_model = load_classifier()
generator_tokenizer, generator_model = load_generator()
label_encoder = load_encoder()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    code_snippet = ''
    flag = ''
    fix = None
    retrieved = []
    if request.method == 'POST':
        code_snippet = request.form['code']
        flag = request.form['flag']
        if code_snippet.strip() and flag.strip():
            prediction, confidence, fix, retrieved = classify_and_fix(
                code_snippet, flag,
                classifier_tokenizer, classifier_model, label_encoder,
                generator_tokenizer, generator_model, top_k=3
            )
    return render_template('index.html', prediction=prediction, confidence=confidence, code=code_snippet, flag=flag, fix=fix, retrieved=retrieved)

if __name__ == '__main__':
    app.run(debug=True)
