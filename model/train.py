from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


def train_model():
    df = datasets.load_iris()

    x = df.data
    y = df.target

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=42)
    dt = DecisionTreeClassifier().fit(xtrain, ytrain)
    prediction = dt.predict(xtest)

    accuracy = accuracy_score(ytest, prediction)
    joblib.dump(dt, 'iris-model.model')
    print('Model have accuracy: {}'.format(accuracy))