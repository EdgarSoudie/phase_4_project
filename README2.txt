# Twitter Sentiment Analysis Project

[cite_start]This project focuses on analyzing public sentiment towards brands and products on Twitter using Natural Language Processing (NLP) techniques. [cite_start]The goal is to classify tweets into positive, negative, or neutral sentiment categories, provide comparative insights between major brands like Apple and Google, and offer actionable recommendations for customer satisfaction.

## Table of Contents
* [1. Business Understanding](#1-business-understanding)
* [2. Data Understanding](#2-data-understanding)
* [3. Data Preparation](#3-data-preparation)
* [4. Modeling](#4-modeling)
* [5. Evaluation](#5-evaluation)
* [6. Deployment & Recommendations](#6-deployment--recommendations)
* [7. Conclusion](#7-conclusion)
* [8. Technologies Used](#8-technologies-used)

## 1. Business Understanding

[cite_start]In today's competitive landscape, understanding customer perceptions is paramount. [cite_start]Social media, particularly platforms like Twitter, offers a rich source of real-time public sentiment. [cite_start]This project aims to leverage Twitter data to gain insights into customer emotions regarding various products and brands.

### Objectives
* [cite_start]To classify tweet sentiments as positive, negative, or neutral.
* [cite_start]To compare sentiments and common word usage for Apple and Google products.
* [cite_start]To build and evaluate binary text classifiers for positive vs. negative emotions.
* [cite_start]To build and evaluate multiclass classifiers for positive, negative, and neutral emotions.
* [cite_start]To improve model performance, especially for minority classes.

## 2. Data Understanding

### Dataset Overview
[cite_start]The dataset for this project is sourced from data.world, containing tweet texts and their associated sentiment labels towards specific brands or products.

[cite_start]Initial Data Overview: The dataset contains three columns: `tweet_text`, `emotion_in_tweet_is_directed_at`, and `is_there_an_emotion_directed_at_a_brand_or_product`.
* [cite_start]`tweet_text`: Contains the raw tweet content.
* [cite_start]`emotion_in_tweet_is_directed_at`: Indicates the specific product/brand the emotion is directed at (e.g., iPhone, Google, iPad). [cite_start]This column had a significant number of missing values (5552 out of 8721).
* [cite_start]`is_there_an_emotion_directed_at_a_brand_or_product`: The target variable, initially having four categories: 'No emotion toward brand or product' (majority class), 'Positive emotion', 'Negative emotion', and 'I can't tell'.

### Exploratory Data Analysis (EDA)
* [cite_start]Sentiment Distribution: Visualizations confirmed significant class imbalance, especially with the 'No emotion toward brand or product' (later 'Neutral emotion') being the dominant category.
* [cite_start]Brand-wise Sentiment: Sentiment distribution for Apple and Google products was analyzed, revealing that Apple had more positive mentions, while Google had a higher proportion of neutral mentions.
* [cite_start]Common Words: Frequency distributions of words were analyzed for positive and negative emotions, as well as for Apple and Google specific tweets, to identify key terms associated with different sentiments and brands.

## 3. Data Preparation

### Data Cleaning Highlights:
* [cite_start]Duplicated rows were identified and removed (22 duplicates).
* [cite_start]Missing values in `tweet_text` (1 instance) were dropped.
* [cite_start]For sentiment classification, 'No emotion toward brand or product' and 'I can't tell' were combined into a 'Neutral emotion' category for multiclass, while only 'Positive emotion' and 'Negative emotion' were used for binary classification.

### Feature Engineering
[cite_start]A robust text preprocessing pipeline was implemented to prepare the tweet data for modeling:
* [cite_start]Lowercase Conversion: All text was converted to lowercase.
* [cite_start]Remove Bracketed Text: Text enclosed in square brackets (e.g., `[link]`) was removed.
* [cite_start]Remove URLs: URLs (`http://`, `https://`, `www.`, `bit.ly/`) were removed.
* [cite_start]Remove Tags & Hashtags: HTML tags (`<.*?>+`) and hashtags (`#\w+`) were removed.
* [cite_start]Remove Alphanumeric Words: Words containing both letters and numbers (e.g., `3G`, `iPad2`) were removed.
* [cite_start]Tokenization: `TweetTokenizer` was used to split text into words, while also stripping Twitter handles (`@mention`).
* [cite_start]Remove Empty Tokens & Filter by Length: Empty tokens were removed, and tokens less than 3 characters long were filtered out.
* [cite_start]Stop Word Removal: Common English stop words (e.g., "the", "is", "a") were removed using NLTK's stopwords list.
* [cite_start]Punctuation Removal: Punctuation tokens were removed.
* [cite_start]Stemming: `PorterStemmer` was applied to reduce words to their root form (e.g., "running" to "run").
* [cite_start]Join Tokens & Normalize Whitespace: Cleaned tokens were joined back into strings.

[cite_start]For feature extraction, `TfidfVectorizer` and `CountVectorizer` were employed to convert the preprocessed text into numerical features for machine learning models.

## 4. Modeling

### Model Building
* [cite_start]**Binary Classification (Positive vs. Negative Emotions)**: The analysis focused on discriminating between positive and negative sentiments.
    * [cite_start]Logistic Regression (Baseline & Tuned): A `Pipeline` with `TfidfVectorizer` and `LogisticRegression` was used. [cite_start]`GridSearchCV` was applied for hyperparameter tuning, and `class_weight='balanced'` was used to address class imbalance.
    * [cite_start]Multinomial Naive Bayes (Tuned & with SMOTE): A `Pipeline` with `CountVectorizer` / `TfidfVectorizer` and `MultinomialNB` was used. [cite_start]`GridSearchCV` was performed, and `SMOTE` was integrated into an `ImbPipeline` to handle imbalance.
* **Multiclass Classification (Positive, Negative, Neutral Emotions)**:
    * [cite_start]Support Vector Classifier (SVC): A `Pipeline` with `TfidfVectorizer` and `SVC` was used. [cite_start]`class_weight='balanced'` and `RandomOverSampler` were explored to mitigate imbalance.
    * [cite_start]K-Nearest Neighbors (KNN): A `Pipeline` with `TfidfVectorizer` and `KNeighborsClassifier` was used, with `RandomOverSampler` for imbalance.
    * [cite_start]Multi-Layer Perceptron (MLP) Classifier: A basic neural network was implemented using `StandardScaler` for feature scaling and `MLPClassifier`. [cite_start]Hyperparameter tuning was performed with `GridSearchCV`.

## 5. Evaluation

### Results & Evaluation
* **Binary Classifiers**
    * Logistic Regression:
        * [cite_start]Baseline: Achieved ~85% accuracy. [cite_start]However, recall for 'Negative emotion' was very low (0.08), indicating bias towards the majority 'Positive emotion' class.
        * [cite_start]Tuned & Balanced: Accuracy slightly reduced to ~83-84%. [cite_start]Crucially, the recall for 'Negative emotion' significantly improved to ~0.58 with precision ~0.50, showing a better balance in handling the minority class.
    * Multinomial Naive Bayes:
        * [cite_start]Tuned: Achieved ~86-87% accuracy. [cite_start]Precision (~0.60) and recall (~0.50) for 'Negative emotion' were better than the baseline Logistic Regression.
        * [cite_start]With SMOTE: Accuracy slightly dropped to ~83%. [cite_start]Precision (~0.50) and recall (~0.51) for 'Negative emotion' achieved a good balance.
    * [cite_start]Key takeaway: The binary classifiers, especially tuned Logistic Regression and Multinomial Naive Bayes with SMOTE, showed reasonable performance for distinguishing positive and negative sentiments, with efforts to balance minority class prediction.
* **Multiclass Classifiers**
    * [cite_start]SVC, KNN, MLP: All multiclass models struggled significantly, with overall accuracies ranging from 52% to 60%. [cite_start]The precision and recall for 'Negative emotion' were particularly weak across these models (e.g., SVC: 0.18 precision, 0.46 recall). [cite_start]The class imbalance, with 'Neutral emotion' being a large majority, heavily biased these models. [cite_start]They found it difficult to define clear boundaries between the three sentiment categories.

## 6. Deployment & Recommendations

* [cite_start]Continuous Sentiment Monitoring: Implement social media strategies for ongoing tracking of public sentiment to support informed business decisions.
* [cite_start]Product/Service Enhancement: Use analyzed sentiment data to directly improve product features and service quality.
* [cite_start]Competitor Analysis: Extend the analysis to include industry competitors for competitive intelligence and identifying unique positioning opportunities.

## 7. Conclusion

[cite_start]The project successfully demonstrated the application of NLP for Twitter sentiment classification. [cite_start]The models developed provide a foundation for companies like Apple and Google to track public sentiment. [cite_start]A key limitation identified is the inherent subjectivity and missing data within the crowd-sourced dataset, which significantly impacted the performance of multiclass classifiers, especially concerning the minority classes.

## 8. Technologies Used
* Python
* [cite_start]Pandas: Data manipulation and analysis.
* [cite_start]NumPy: Numerical operations.
* [cite_start]Matplotlib, Seaborn: Data visualization.
* [cite_start]NLTK: Natural Language Toolkit for text preprocessing (tokenization, stopwords, stemming, VADER).
* [cite_start]Scikit-learn: Machine learning models (Logistic Regression, Multinomial Naive Bayes, SVC, KNN, MLPClassifier), feature extraction (TfidfVectorizer, CountVectorizer), model selection (GridSearchCV), and evaluation metrics (classification_report, confusion_matrix, roc_curve, accuracy_score).
* [cite_start]Imblearn: For handling imbalanced datasets (SMOTE, RandomOverSampler, ImbPipeline).