import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier


def train_model(model_name, data, label, cfg):
    if model_name == 'lr':
        model = LogisticRegression(C=cfg.C)
    elif model_name == 'knn':
        model = KNeighborsClassifier(n_neighbors=cfg.n_neighbors)
    elif model_name == 'svm':
    	model = SVC(C=cfg.C)
    elif model_name == 'adaboost':
        model = AdaBoostClassifier(n_estimators=cfg.n_estimators)
    data = np.reshape(data, (data.shape[0], -1))
    model.fit(data, label)
    return model


def eval_model(model, data):
    data = np.reshape(data, (data.shape[0], -1))
    predict = model.predict(data)
    return predict


def compute_accuracy(predict, label):
    num_correct = np.sum(predict == label)
    accuracy = float(num_correct) / len(label)
    return accuracy

