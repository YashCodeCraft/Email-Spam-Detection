# Email Spam Detection

This project detects whether an email is **spam** or **ham** using Natural Language Processing (NLP) and Machine Learning techniques.

## Project Overview

1. **Data**: 
   - The data comes from the Enron email dataset, with separate folders for `ham` (non-spam) and `spam` emails.
   
2. **Steps**:
   - Load the emails from the directories.
   - Preprocess the text using **SpaCy** to lemmatize and remove stopwords/punctuation.
   - Convert text data to **TF-IDF** vectors.
   - Train a **Naive Bayes classifier** to classify the emails.
   - Test the model and evaluate its performance using metrics like precision, recall, and F1-score.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/YashCodeCraft/Email-Spam-Detection.git
   cd Email-Spam-Detection
   ```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Download the en_core_web_sm model for SpaCy:

```bash
python -m spacy download en_core_web_sm
```
## How to Run
1. Place the Enron email dataset in the respective directories:

- `ham/`
- `spam/`
2. Run the script to preprocess the data, train the model, and evaluate:

```bash
python email_spam_detection.py
```
3. Predict a single email:

```python
new_email = "Your free lottery ticket is waiting!"
print(predict_email(new_email))
```
## Results
Model evaluation metrics (Precision, Recall, F1-score) will be printed during the process.
