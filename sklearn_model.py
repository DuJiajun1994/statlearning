import pickle
import numpy as np
from data_provider import DataProvider
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


def train_model(model_name, data, label):
    if model_name == 'lr':
        model = LogisticRegression()
    elif model_name == 'knn':
        model = KNeighborsClassifier()
    elif model_name == 'random_forest':
        model = RandomForestClassifier()
    elif model_name == 'adaboost':
        model = AdaBoostClassifier()
    data = np.reshape(data, (data.shape[0], -1))
    model.fit(data, label)
    with open('output/{}.pkl'.format(model_name), 'wb') as fid:
        pickle.dump(model, fid)


def eval_model(model_name, data, label):
    with open('output/{}.pkl'.format(model_name), 'rb') as fid:
        model = pickle.load(fid)
    data = np.reshape(data, (data.shape[0], -1))
    predict = model.predict(data)
    num_correct = np.sum(predict == label)
    accuracy = float(num_correct) / len(label)
    print('accuracy: {}'.format(accuracy))


def cross_validation(model_name, provider):
    for i in range(5):
        train_data, train_label, val_data, val_label = provider.get_cv_data(i)
        train_model(model_name, train_data, train_label)
        eval_model(model_name, val_data, val_label)

if __name__ == '__main__':
    provider = DataProvider()
    cross_validation('lr', provider)
