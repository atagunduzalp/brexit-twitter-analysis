import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE 
import pickle

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import seaborn as sns
import matplotlib.pyplot as plt

AFTER_NORMALIZATION_CSV = "tweets-after-normalization.csv"

def read_file_from_csv(AFTER_NORMALIZATION_CSV):
    
    data = pd.read_csv(AFTER_NORMALIZATION_CSV)
    word_cloud(data)
    
    want_to_exit = data[(data['sentiment'] == 1)]
    want_to_stay = data[(data['sentiment'] == 0)]
    notr = data[(data['sentiment'] == 2)]
    draw_histogram(data, want_to_exit, want_to_stay, notr)
    
    kfol_cross_validation(data, want_to_exit, want_to_stay, notr)

def draw_histogram(data, want_to_exit, want_to_stay, notr):

    labels = 'Want to Exit', 'Want to Stay', 'Notr'
    sizes = [len(want_to_exit), len(want_to_stay), len(notr)]
    colors = ['yellowgreen', 'lightcoral', 'blue']
    explode = (0.1, 0.1, 0.1)  # explode 1st slice

    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
    autopct='%1.1f%%', shadow=True, startangle=140)

    plt.axis('equal')
    plt.show()

def word_cloud(data):
    text = " ".join(tweet for tweet in data['tweet'])
    mask = np.array(Image.open("uk.png"))
    wordcloud = WordCloud(max_words=5000, background_color="white", mask=mask).generate(text)
    image_colors = ImageColorGenerator(mask)

    plt.figure(figsize=[4,4])
    plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
    plt.axis("off")
    plt.show()

def kfol_cross_validation(data, want_to_exit, want_to_stay, notr):    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    tweet_df = data['tweet']
    sentiment_df = data['sentiment']
    vectorizer= CountVectorizer(min_df=0, lowercase=False)
    BEST_SCORE = 0.0
    for train_ix, test_ix in kfold.split(tweet_df, sentiment_df):
        X_train, X_test = tweet_df[train_ix], tweet_df[test_ix]
        y_train, y_test = sentiment_df[train_ix], sentiment_df [test_ix]
        
        BEST_SCORE = random_forest(X_train, y_train, X_test, y_test, vectorizer, BEST_SCORE, want_to_exit, want_to_stay, notr)
        logistic_regression(X_train, y_train, X_test, y_test, vectorizer)

def logistic_regression(trainingTweets, trainingLabels, testTweets,testLabels, vectorizer):
    vectorizer.fit(trainingTweets)
    X_train = vectorizer.transform(trainingTweets)
    X_test  = vectorizer.transform(testTweets)
    
    penalty = ['l1', 'l2']
    C = np.logspace(-4,4,20)
    solver = ['newton-cg', 'liblinear']
    hyperparameters = dict(penalty=penalty, C=C, solver=solver)
    logreg = LogisticRegression()
    clf = GridSearchCV(logreg, hyperparameters, cv=10)
    
    oversample = SMOTE()
    x_train_res, y_train_res = oversample.fit_sample(X_train, trainingLabels)
    #Fitting Model
    classifier = clf.fit(x_train_res, y_train_res)
    print('Best Penalty:', classifier.best_estimator_.get_params()['penalty'])
    print('Best C:', classifier.best_estimator_.get_params()['C'])
    print('Best solver:', classifier.best_estimator_.get_params()['solver'])
    
    y_pred = classifier.predict(X_test)
    print("LOGISTIC REGRESSION: ")
    conf_matrix  = confusion_matrix(y_pred, testLabels)
    print(conf_matrix)
    show_confusion_matrix(conf_matrix)
    print(classification_report(testLabels, y_pred))
    print("logistic_regression score: " + str(accuracy_score(y_pred, testLabels)))

def random_forest(trainingTweets, trainingLabels, testTweets,testLabels, vectorizer, BEST_SCORE, want_to_exit, want_to_stay, notr):

    vectorizer.fit(trainingTweets)
    X_train = vectorizer.transform(trainingTweets)
    X_test  = vectorizer.transform(testTweets)
    oversample = SMOTE()
    x_train_res, y_train_res = oversample.fit_sample(X_train, trainingLabels)
    # select_hyperparameters(X_train, trainingLabels)
    
    classifier = RandomForestClassifier(n_estimators= 336, min_samples_split= 5, min_samples_leaf= 1, max_features= 'log2', 
                                    max_depth= 70, criterion= 'gini', bootstrap= False)

    classifier.fit(x_train_res, y_train_res)
    y_pred = classifier.predict(X_test)
    conf_matrix = confusion_matrix(y_pred, testLabels )
    print(conf_matrix)
    show_confusion_matrix(conf_matrix)

    print('Random Forest Report:\n' + str(classification_report(testLabels, y_pred)))
    score = accuracy_score(y_pred, testLabels)
    print("random_forest scor : " + str(accuracy_score(y_pred, testLabels)))
    if score > BEST_SCORE:
        BEST_SCORE = score
        filename = 'finalized_model.sav'
        pickle.dump(classifier, open(filename, 'wb'))
        pickle.dump(vectorizer, open('count_vect', 'wb'))
    return BEST_SCORE

def show_confusion_matrix(confusion_matrix):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(confusion_matrix)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Stay', 'Predicted Exit'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual Stay', 'Actual Exit'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='red')
    plt.show()

def select_hyperparameters(X, y):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 300, stop = 1500, num = 100)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt', 'log2']
    # criterion{“gini”, “entropy”}, default=”gini”
    criterion = ['gini', 'entropy']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    # max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'criterion': criterion,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    rf = RandomForestClassifier(class_weight='balanced')
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=8, n_jobs = -1)
    rf_random.fit(X, y)
    best_params = rf_random.best_params_
    print("best results: " + str(rf_random.best_params_))
    # best results: {'n_estimators': 336, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 70, 'criterion': 'gini', 'bootstrap': False}

# if __name__ == "__main__":
#     read_file_from_csv(AFTER_NORMALIZATION_CSV)
