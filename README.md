# SVACE Vulnerability Classifier

A machine learning-based vulnerability detection system that classifies code snippets into different categories based on static analysis warnings.

## Features

- **Multi-category Classification**: Classifies code into 5 categories:
  - Confirmed
  - False Positive
  - Won't Fix
  - Pending
  - Undecided

- **Issue Type Detection**: Handles different types of code issues:
  - Null dereference
  - Allocation mismatches
  - Invalid comparisons

- **Web Interface**: User-friendly Flask web application for easy interaction
- **Confidence Scoring**: Provides confidence scores for predictions
- **CodeBERT-based**: Uses Microsoft's CodeBERT model for code understanding

## Project Structure

```
svace_classifier/
├── app.py                 # Flask web application
├── train_model.py         # Model training script
├── model_utils.py         # Model utility functions
├── requirements.txt       # Python dependencies
├── data/
│   └── svace_dataset.csv  # Training dataset
├── templates/
│   └── index.html         # Web interface template
├── saved_model/           # Trained model files (generated)
└── label_encoder.joblib   # Label encoder (generated)
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd svace_classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python train_model.py
   ```

4. **Run the web application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to `http://127.0.0.1:5000`

## Usage

1. **Select Issue Type**: Choose from the dropdown menu:
   - Null dereference
   - Allocation mismatches
   - Invalid comparisons

2. **Enter Code Snippet**: Paste your code in the text area

3. **Analyze**: Click the "Analyze" button to get predictions

4. **View Results**: See the classification result and confidence score

## Dataset Format

The training dataset (`data/svace_dataset.csv`) should have the following columns:
- `code`: The code snippet (properly quoted)
- `flag`: The type of issue detected
- `label`: The classification category

Example:
```csv
code,flag,label
"int *ptr = NULL; *ptr = 42;",Null dereference,Confirmed
"int a = 5; if(a = 10) {}",Invalid comparisons,False Positive
```

## Model Details

- **Base Model**: Microsoft CodeBERT
- **Architecture**: RoBERTa-based sequence classification
- **Input Format**: `[flag] code` (concatenated)
- **Output**: 5-class classification with confidence scores

## Dependencies

- transformers==4.41.1
- torch>=2.0.0
- pandas==2.2.2
- scikit-learn==1.5.0
- joblib==1.4.2
- flask==3.0.3

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Microsoft CodeBERT for the base model
- Hugging Face Transformers library
- Flask web framework 