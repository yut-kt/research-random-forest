# -*- coding: utf-8 -*-

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_svmlight_file


def main():
    trains, lables = load_svmlight_file('./data/svm_train')
    clf = RandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=None, min_samples_split=10,
                                 max_features=10, bootstrap=False, n_jobs=-1)

    clf.fit(trains, lables)
    joblib.dump(clf, 'model/random-forest.pkl')


if __name__ == '__main__':
    main()
