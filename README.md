# Vulnerability Classifier with RAG and Code Generation

A comprehensive machine learning-based vulnerability detection and code fixing system that combines classification, retrieval-augmented generation (RAG), and multi-task learning (MCP) to analyze and fix code vulnerabilities.

## 🚀 Features

### **Multi-Category Classification**
- Classifies code snippets into 5 categories:
  - **Confirmed** - Actual vulnerability detected
  - **False Positive** - Incorrect warning
  - **Won't Fix** - Known issue, intentionally not fixed
  - **Pending** - Requires further investigation
  - **Undecided** - Unable to determine

### **Issue Type Detection**
- Handles multiple types of code issues:
  - **Null dereference** - Accessing null pointers
  - **Allocation mismatches** - Memory allocation/deallocation issues
  - **Invalid comparisons** - Logical comparison errors

### **Advanced Code Generation**
- **RAG (Retrieval-Augmented Generation)**: Combines similar examples with generative AI
- **Code Fix Suggestions**: Provides specific code fixes for detected vulnerabilities
- **Similar Examples**: Shows relevant examples from the training dataset
- **Confidence Scoring**: Provides confidence scores for both classification and fixes

### **Web Interface**
- User-friendly Flask web application
- Real-time code analysis and fixing
- Interactive issue type selection
- Collapsible sections for detailed results

## 🏗️ Architecture

### **Multi-Model System**
- **CodeBERT Classifier**: Microsoft's CodeBERT for vulnerability classification
- **T5 Generator**: Text-to-text model for code fix generation
- **FAISS Index**: Efficient similarity search for RAG
- **Sentence Transformers**: Embedding generation for retrieval

### **RAG Pipeline**
1. **Input Processing**: Combines issue type and code snippet
2. **Classification**: Determines vulnerability category and confidence
3. **Retrieval**: Finds similar examples using FAISS
4. **Generation**: Generates code fixes using retrieved context
5. **Output**: Returns classification, confidence, fixes, and examples

## 📁 Project Structure

```
SVACE/
├── app.py                    # Main Flask web application
├── model_utils.py            # Model loading and prediction utilities
├── train_classifier.py       # Classification model training script
├── train_generator.py        # Code generation model training script
├── retrieval_utils.py        # FAISS index building and querying
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore               # Git ignore rules
├── data/
│   └── svace_dataset.csv    # Training dataset with code fixes
└── templates/
    └── index.html           # Web interface template
```

## 🛠️ Installation

### **Prerequisites**
- Python 3.8+
- Git

### **Setup Instructions**

1. **Clone the repository**
   ```bash
   git clone https://github.com/iamrinal/SVACE.git
   cd SVACE
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv svace_env
   # On Windows:
   svace_env\Scripts\activate
   # On macOS/Linux:
   source svace_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the models**
   ```bash
   # Train the classification model
   python train_classifier.py
   
   # Train the code generation model
   python train_generator.py
   
   # Build the FAISS index for retrieval
   python retrieval_utils.py
   ```

5. **Run the web application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   Navigate to `http://127.0.0.1:5000`

## 📖 Usage

### **Web Interface**

1. **Select Issue Type**: Choose from the dropdown:
   - Null dereference
   - Allocation mismatches
   - Invalid comparisons

2. **Enter Code Snippet**: Paste your vulnerable code in the text area

3. **Analyze**: Click "Analyze" to get comprehensive results

4. **Review Results**: View:
   - **Classification**: Vulnerability category and confidence
   - **Suggested Code Fix**: Generated fix for the issue
   - **Similar Examples**: Relevant examples from the dataset

### **Example Input**
```c
// Issue Type: Null dereference
int *ptr = NULL;
*ptr = 42;  // This will cause a segmentation fault
```

### **Example Output**
- **Classification**: Confirmed (95% confidence)
- **Suggested Fix**: 
  ```c
  int *ptr = malloc(sizeof(int));
  if (ptr != NULL) {
      *ptr = 42;
      free(ptr);
  }
  ```
- **Similar Examples**: Shows related null dereference cases

## 📊 Dataset Format

The training dataset (`data/svace_dataset.csv`) includes:

| Column | Description | Example |
|--------|-------------|---------|
| `code_with_bug` | Vulnerable code snippet | `"int *ptr = NULL; *ptr = 42;"` |
| `flag` | Issue type | `"Null dereference"` |
| `label` | Classification category | `"Confirmed"` |
| `fixed_code` | Corrected code | `"int *ptr = malloc(sizeof(int)); if(ptr) *ptr = 42;"` |

## 🤖 Model Details

### **Classification Model**
- **Base Model**: Microsoft CodeBERT (`microsoft/codebert-base`)
- **Architecture**: RoBERTa-based sequence classification
- **Input Format**: `[flag] code` (concatenated)
- **Output**: 5-class classification with confidence scores

### **Generation Model**
- **Base Model**: T5 (`t5-base`)
- **Architecture**: Text-to-text transformer
- **Input Format**: `"fix: [flag] code"`
- **Output**: Fixed code snippet

### **Retrieval System**
- **Embedding Model**: Sentence Transformers with CodeBERT
- **Index**: FAISS for efficient similarity search
- **Top-k**: Returns 3 most similar examples

## 📦 Dependencies

```
transformers==4.54.0
torch>=2.0.0
pandas==2.2.2
scikit-learn==1.5.0
joblib==1.4.2
flask==3.0.3
faiss-cpu==1.7.4
sentence-transformers==2.6.1
tqdm==4.66.1
```

## 🔧 Training

### **Classification Training**
```bash
python train_classifier.py
```
- Uses CodeBERT for sequence classification
- Trains on `[flag] code` → `label` mapping
- Saves model to `saved_classifier/`

### **Generation Training**
```bash
python train_generator.py
```
- Uses T5 for text-to-text generation
- Trains on `"fix: [flag] code"` → `fixed_code` mapping
- Saves model to `saved_generator/`

### **Index Building**
```bash
python retrieval_utils.py
```
- Builds FAISS index for similarity search
- Saves index to `faiss_index/`

## 🧪 Testing

### **Sample Code Snippets to Test**

1. **Null Dereference**:
   ```c
   char *str = NULL;
   strcpy(str, "hello");
   ```

2. **Allocation Mismatch**:
   ```c
   int *arr = malloc(10 * sizeof(int));
   free(arr);
   arr[5] = 42;  // Use after free
   ```

3. **Invalid Comparison**:
   ```c
   int a = 5;
   if (a = 10) {  // Assignment instead of comparison
       printf("a is 10");
   }
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Microsoft CodeBERT** for code understanding
- **Hugging Face Transformers** for model infrastructure
- **FAISS** for efficient similarity search
- **Flask** for web framework
- **Sentence Transformers** for embeddings

## 📞 Support

For issues and questions:
1. Check the [Issues](https://github.com/iamrinal/SVACE/issues) page
2. Create a new issue with detailed description
3. Include code snippets and error messages

---

**Made with ❤️ for the security community** 
