Background Information

Yelp is a platform that helps people find great local businesses such as dentists, resturants, haircuts and mechanics.
As listed on Yelp official website, by Q1 2018, Yelp has already had 70 million unique visitors and over 155 million reviews written by customers. With such big amount of data, we are able to conduct ETL, feature engineering, and modelling using data analytics techniques to see if we can get any useful and interesting results that can make business sense.
Yelp dataset challenge is a chance for students to conduct research or analysis on Yelp data and share their discoveries with Yelp. Round 11 of this challenge started in January 18, 2018 and will run through June 30, 2018. With raw dataset provided by Yelp open data source, further technical work can be done.



Process and Methods Used

In this project, four sections of analysis are going to be done.
The first section is Data Preprocessing. Raw text dataset in Json format will be transferred to pandas data table for further analysis in Python. Also, as raw dataset is very large, filters will be used to narrow down the size of data. Newly cleaned and aggregated datasets will be saved in CSV format.
Section 2 is Natural Language Processing for restaurants review data. Tokenization with stemming and lemmatization will be adopted to convert user review text data to vectors for NLP study. In addition, a ‘Similar Review Search Engine’ will be built to help users find similar restaurants based on reviews these merchants get. Classification models such as Naïve-Bayes, Logistic Regression and Random Forest will be conducted to classify positive and negative restaurants.
Section 3 is Clustering and PCA. Similar restaurants can be clustered as a group using classification algorithms like K-Means. By this way, we are able to analyze what elements does a ‘good’ or ‘positive’ restaurant need to have. PCA (Principle Components Analysis) will also be conducted in this case to reduce dimension of data.
The last section is Restaurant Recommender. Content based recommender, item-item based recommender, user-user based recommender, and popularity based recommender are going to be used in the Yelp dataset challenge.
