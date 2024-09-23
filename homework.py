import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

#Help function 1:
#open files that have format - number, then tab, then text 
# that the training file opener
def train_file_reader(filepath):
    scores = []
    reviews = []
    
    with open(filepath, 'r') as file:
        for line in file:
            # the format of the file is that every row has score, then tab, then the review.
            # we want to split the score and review using split function wuth tab argument
            score, review = line.split('\t')
            scores.append(int(score))
            reviews.append(review)
            
    return scores, reviews

#Help function 2:
#open files that have format - just text, each line is different review
def read_test_dat_file(filepath):
    data = []
    
    with open(filepath, 'r') as file:
        for line in file:
            data.append(line)  
            
    return data


#reading the training file
scores, reviews = train_file_reader('path/train_file.dat')


# Creeating data frame as it is easy to work with using library pandas
df = pd.DataFrame({
    'score': scores,
    'review': reviews
})

# creating custom version of Tfidf using the optional paramethers
vectorizer = TfidfVectorizer(
    stop_words='english',         
    lowercase=True,
    ngram_range=(1,3),          
    max_features=20000,            
    min_df=2,                    
    max_df=0.8,                   
    sublinear_tf=True,            
    norm='l2'                     
)


# transform the text to a matrix
# every row is a review and every colomn is a word, the cell shows the TF-IDF score 
X = vectorizer.fit_transform(df['review'])  

Y = df['score']
#X and Y are ready to train my model

# Train logistic regression model
model = LogisticRegression()
model.fit(X, Y)

# Read test file - it is only reviews(text)
test_texts = read_test_dat_file('path/test.dat')

# Vectorize the test data using the same vectorizer used for training
X_test = vectorizer.transform(test_texts)

#Prediction results
test_predictions = model.predict(X_test)

# Saving the results of my prediction to a .dat file
with open('result.dat', 'w') as f:
    for prediction in test_predictions:
        f.write(f'{prediction}\n')

