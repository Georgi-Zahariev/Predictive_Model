# Predictive_Model
Using Logistic Regression model the program predicts whether a review is positive ot not. 

Project for CS484 - Data Mining at GMU. May not be used from other students!

The project wants us to create a predictive model using logistic regression. I am using Python to code it and I am importing libraries such as pandas, sklearn (scikit-learn). We have training and test files provided to us. Training data consists of reviews and +1 if a positive review, and -1 if negative. Test data has only reviews. Our goal is to predict if these reviews are positive(+1) or negative(-1).

Importing Pandas library I use the following command in my code:
•	import pandas as pd
If needed installation run the command: pip install pandas

When I am importing everything I need from scikit-learn I use the following commands in my code:

Importing TfidfVectorizer from sklearn
•	from sklearn.feature_extraction.text import TfidfVectorizer

Importing LogisticRegression from sklearn
•	from sklearn.linear_model import LogisticRegression

If needed installation run the command: pip install scikit-learn

The steps that have been taken are:
1.	Extracting the data.
2.	Setting the data in a Data Frame.
3.	TF-IDF vectorizer creation and setting up. It will be used to transform the data.
4.	Using vectorizer.fit_transform(…) on my review data from the training file the following things happen
•	Preprocessing the data from text to more suitable data.
•	Creation of a matrix that has easy review as a row and each word as a column and the cell is a numerical value.
•	Removing stop_words, making all of them lowercase, creating n-grams, eliminating words in less than 2 sources, eliminating words that are in more than 80% of the sources, and implementing the sublinear_tf that makes the TF scaling logarithmic and norm=”l2” for normalization. These are functionalities that my TfidfVectorizer has and applies to the data passed to him.
5.	Initiating the logistic regression model.
6.	Training the logistic regression model with the train set after the changes above have been implemented.
7.	Using the same TF-IDF vectorizer, with the same functionalities explained above with my test data.
8.	Logistic Regression Model makes the predictions based on the test data that was made suitable for it.
9.	Creating a file from the results I get from my prediction. 

Important for running the code.

There are two data extraction. The initial one is when extracting the train data and the second one is when extracting the test data. These are the commands in my source code and they are implemented with my specific source path. When testing source path needs to be adapted to your local machine.

scores, reviews = train_file_reader('path/train_file.dat')
Please, change the source path to the proper one on your machine.

test_texts = read_test_dat_file('path/test.dat')
Please, change the source path to the proper one on your machine.



![image](https://github.com/user-attachments/assets/e1c31a42-b0a6-46e9-bf43-2d7970650aee)
