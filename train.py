# -*- coding: utf-8 -*-

import numpy
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier


def main():
    dataset = numpy.load('data/dataset.npz')

    clf = RandomForestClassifier(n_estimators=20, criterion='entropy', max_depth=None, min_samples_split=10,
                                 max_features=10, bootstrap=False, n_jobs=-1)

    clf.fit(dataset['trains'], dataset['labels'])
    joblib.dump(clf, 'model/random-forest.pkl')


if __name__ == '__main__':
    main()
