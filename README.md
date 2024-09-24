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
   git clone https://github.com/your-username/email-spam-detection.git
   cd email-spam-detection
