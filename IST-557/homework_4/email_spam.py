import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.utils import resample
from sklearn.naive_bayes import MultinomialNB

# read the training data
data = pd.read_csv("./data_train_hw4_problem1.csv", encoding="latin-1")

# label spam as 1 and not-spam as 0
data.spam = data.spam.apply(lambda x:1 if x == True else 0)

# train-test split
x_train, x_val, y_train, y_val = train_test_split(data.text, data.spam, test_size=0.001, random_state=0)

x_train_list = x_train.tolist()
vectorizer = TfidfVectorizer(
    input= x_train_list,
    stop_words='english'
)
train_features = vectorizer.fit_transform(x_train_list)
val_features = vectorizer.transform(x_val)

model = MultinomialNB()
cv_results = cross_validate(
    model, train_features, y_train, scoring='balanced_accuracy', return_train_score= True, return_estimator= True
)

scores = []
for fold_id, cv_model in enumerate(cv_results["estimator"]):
    scores.append(balanced_accuracy_score(y_val, cv_model.predict(val_features)))


predicted = cv_results["estimator"][np.argmax(scores)].predict(val_features)
actual = y_val.tolist()
results = confusion_matrix(actual, predicted)

test_data = pd.read_csv("./data_test_hw4_problem1.csv", encoding="latin-1")
test_data = np.array(test_data.text)
test_features = vectorizer.transform(test_data)
test_results = cv_results["estimator"][np.argmax(scores)].predict(test_features)

test_results = pd.DataFrame(test_results,columns=["spam"])
test_results.spam = test_results.spam.apply(lambda x:"TRUE" if x == 1 else "FALSE")

test_results.to_csv("./predictions.csv")



