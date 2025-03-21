# SENTIMENT-ANALYSIS

*COMPANY*: CODTECH IT SOLUTION

*NAME*: MADHAN G

*INTERN ID*: CT08VCD

*DOMAIN*: DATA ANALYTICS

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH


# PERFORM SENTIMENT ANALYSIS ON TEXTUAL DATA (E.G., TWEETS) USING NATURAL LANGUAGE PROCESSING (NLP) TECHNIQUES.

**Sentiment analysis** is the process of determining the emotional tone behind a piece of text. It aims to identify whether the expressed opinion is positive, negative, or neutral. Essentially, it's about understanding how people feel about something from their written or spoken words.

*STEPS:*
1. Downloads the Sentiment140 dataset from Kaggle.
2. Preprocesses the data by removing stop words and stemming.
3. Splits the data into training and testing sets.
4. Converts the text data into numerical data using TF-IDF vectorization.
5. Trains a logistic regression model on the training data.
6. Evaluates the model on the testing data.
7. Saves the trained model to a file.

*PREREQUISITES*
1. Make sure you have the necessary libraries installed (e.g., pandas, scikit-learn, nltk).
2. Download the 'kaggle.json' file from your Kaggle account and place it in the current directory.
3. Run the script.

 **Import necessary libraries**
 
          import numpy as np
          import pandas as pd
          import re
          import nltk
          from nltk.corpus import stopwords
          from nltk.stem.porter import PorterStemmer
          from sklearn.feature_extraction.text import TfidfVectorizer
          from sklearn.model_selection import train_test_split
          from sklearn.linear_model import LogisticRegression
          from sklearn.metrics import accuracy_score
          import pickle

**Download and extract the dataset**

            !pip install kaggle
            mkdir -p ~/.kaggle
            !cp kaggle.json ~/.kaggle/
            !chmod 600 ~/.kaggle/kaggle.json
            !kaggle datasets download -d kazanova/sentiment140
            
            from zipfile import ZipFile
            dataset = '/content/sentiment140.zip'
            
            with ZipFile(dataset,'r') as zip:
            zip.extractall()
            print('The dataset is extracted')

**Stopword removing**
Stop word removal in text preprocessing involves eliminating common, high-frequency words (like "the," "a," "is") that add little meaning. This reduces noise, improves efficiency, and enhances model performance by focusing on relevant content words.
            
            nltk.download('stopwords')

**Load the dataset**
            
            twitter_data = pd.read_csv('/content/training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1')

**Preprocess the data**
Twitter data is often noisy, with informal language, abbreviations, and misspellings.
Preprocessing improves the accuracy of sentiment analysis, topic modeling, and other NLP tasks.
It helps to focus on relevant information and reduce computational load.

          def stemming(content):
            """Stems and cleans the text data."""
            stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
            stemmed_content = stemmed_content.lower()
            stemmed_content = stemmed_content.split()
            stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
            stemmed_content = ' '.join(stemmed_content)
            return stemmed_content
            
          port_stem = PorterStemmer()
          twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)

**Split the data into training and testing sets**
          
          X = twitter_data['stemmed_content'].values
          Y = twitter_data['target'].values
          X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

**Convert text data to numerical data using TF-IDF vectorization**
A **vectorizer** in NLP transforms text into numerical vectors. Machine learning models need numbers, not raw text, to function. It creates a mathematical representation of words or documents. This enables tasks like sentiment analysis and text classification.
  
          vectorizer = TfidfVectorizer()
          X_train = vectorizer.fit_transform(X_train)
          X_test = vectorizer.transform(X_test)

**Train the logistic regression model**
Logistic regression is a statistical model used for binary classification.
It's computationally efficient, making it suitable for large datasets.   
Training is fast compared to more complex models
It's inherently designed for binary classification problems (e.g. positive vs. negative sentiment).
          
          model = LogisticRegression(max_iter=1000)
          model.fit(X_train, Y_train)

**Evaluate the model**
Model evaluation is the process of assessing how well a machine learning model performs on a given dataset.

          
          X_train_prediction = model.predict(X_train)
          training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
          print('Accuracy on training data : ', training_data_accuracy)
  
          X_test_prediction = model.predict(X_test)
          testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
          print('Accuracy on testing data : ', testing_data_accuracy)

**Save the trained model**
The **pickle module** in Python is used for these primary reasons:
**Object Persistence:** To save complex Python objects (like trained machine learning models, dictionaries, lists) to a file, allowing you to load and reuse them later without recreating them.
**Data Serialization:** To convert Python objects into a byte stream, enabling storage or transmission of data. This is useful for saving the state of an application or sending data over a network.
          
          filename = "trained_model.sav"
          pickle.dump(model, open(filename, 'wb'))

DATASET LINK: https://www.kaggle.com/datasets/kazanova/sentiment140
