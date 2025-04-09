# ✈️ Twitter US Airline Sentiment Analysis

A Natural Language Processing (NLP) project focused on analyzing and classifying public sentiment from tweets directed at major U.S. airlines. This project explores the process of preprocessing tweets, extracting features using vectorization methods, and training classification models to understand public opinion in real time.

---

## 📌 Problem Statement

Airlines often face public scrutiny on platforms like Twitter. This project aims to:
- Classify tweets into **positive**, **neutral**, or **negative** sentiments.
- Understand reasons for negative sentiment (e.g., delays, customer service, baggage issues).

The dataset was sourced from Kaggle and contains tweets from **February 2015** about major U.S. airlines.

---

## 📁 Dataset Overview

- **Source**: [Kaggle - Twitter US Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)
- **Key Columns Used**:
  - `text`: The tweet content.
  - `airline_sentiment`: Sentiment label (positive, neutral, negative).

---

## 🧠 Project Workflow

### 1. 🔍 Data Exploration
- Loaded data from `Tweets.csv`
- Retained relevant columns: `text` and `airline_sentiment`
- Explored class distribution and tweet samples

### 2. 🧼 Text Preprocessing
Performed several preprocessing steps including:
- HTML tag removal
- Tokenization
- Removing numbers and special characters
- Converting to lowercase
- Lemmatization
- Rejoining tokens into cleaned text

### 3. 🧮 Feature Engineering
- Applied **Bag of Words** (CountVectorizer)
- Applied **TF-IDF Vectorizer**

### 4. 🏗️ Model Building
Trained and evaluated the following classifiers using both vectorization methods:
- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)

### 5. 📊 Model Evaluation
- Accuracy scores
- Classification reports
- Confusion matrices
- Visual analysis of sentiment distributions

---

## 📊 Results Summary

| Vectorizer     | Model               | Accuracy |
|----------------|---------------------|----------|
| CountVectorizer| Logistic Regression | ~85%     |
| TF-IDF         | Naive Bayes         | ~82%     |
| TF-IDF         | SVM (Linear Kernel) | ~86%     |

---

## 🔍 Insights

- **Negative sentiments** dominated the dataset.
- Common negative reasons: flight delays, poor customer service, lost baggage.
- **TF-IDF + SVM** provided the best performance.

---

## 🧰 Tech Stack

- Python (Pandas, Numpy, Matplotlib, Seaborn)
- Scikit-learn
- NLTK / spaCy for preprocessing
- Jupyter Notebook

---

## 📁 Project Structure

├── Twitter_US_Airline_Sentiment_Analysis_Project.ipynb # Jupyter Notebook 

├── Twitter_US_Airline_Sentiment_Analysis_Project.html # HTML export 

├── Tweets.csv # Raw dataset 

├── Problem Statement - Twitter US Airline Sentiment.pdf # Provided brief 

├── README.md # Project documentation

yaml
Copy
Edit

---

## 📚 Learning Outcomes

- End-to-end NLP workflow from raw text to classification
- Hands-on experience with CountVectorizer & TF-IDF
- Evaluating and comparing machine learning classifiers
- Real-world application of sentiment analysis in aviation
---

## 📚 Additional Resources

![python Code](https://github.com/Softechanalytics/NLP/blob/main/Twitter_US_Airline_Sentiment_Analysis_Project(Anyakwu_Chukwuemeka_Isaac).ipynb)

![Dataset](

---

## 📬 Author

**Anyakwu Chukwuemeka Isaac**  
Let's connect on [LinkedIn](https://www.linkedin.com/in/chukwuemekaanyakwu2409) or [Email](chuks.isaac70@gmail.com)

---

## 📝 License

For academic and educational use only.  
© Great Learning. All Rights Reserved.
