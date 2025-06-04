from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def get_acc_classifier(x, y, classifier="linear"):
    if classifier=="linear":
        clf = LogisticRegression()
    elif classifier=="tree":
        clf = DecisionTreeClassifier(max_depth=5)
    elif classifier=="random_forest":
        clf = RandomForestClassifier(n_estimators=100, max_depth=5)
    elif classifier=="mlp":
        clf = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=500)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return accuracy_score(y_test, y_pred)