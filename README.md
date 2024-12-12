# Hate-Speech-Detection
# Hate Speech Detection Using Machine Learning  

This repository contains a Python-based machine learning project aimed at detecting hate speech and offensive language in social media posts. The model categorizes text into three classes:  
1. **Hate Speech**  
2. **Offensive Language**  
3. **No Hate or Offensive Language**  

### Key Features:  
- **Dataset:** Utilizes a labeled dataset for training and testing.  
- **Text Preprocessing:** Includes data cleaning steps such as:  
  - Lowercasing text  
  - Removing URLs, special characters, numbers, and punctuation  
  - Removing stopwords and applying stemming for text normalization  
- **Machine Learning Model:** Implements a Decision Tree Classifier for text classification.  
- **Vectorization:** Converts text data into numerical format using CountVectorizer.  
- **Evaluation Metrics:**  
  - Confusion matrix visualization using Seaborn  
  - Model accuracy calculation  

### Requirements:  
- Python  
- NumPy  
- Pandas  
- NLTK  
- scikit-learn  
- Seaborn  
- Matplotlib  

### Usage:  
1. Preprocess text data using the `clean_data` function.  
2. Train the Decision Tree Classifier with the preprocessed dataset.  
3. Use the trained model to classify new text samples into one of the three categories.  

### Example:  
Input text:  
`"Let's unite and kill all the people protesting against the government"`  

Output prediction:  
`"Hate Speech"`  

This repository serves as a foundational project for text classification tasks and can be extended or improved using advanced NLP techniques like TF-IDF, word embeddings, or deep learning models.  

Feel free to fork and contribute! 🎉  
