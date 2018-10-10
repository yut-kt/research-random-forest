# -*- coding: utf-8 -*-

from feature import Feature
from sklearn.externals import joblib


def main():
    feature = Feature()

    tests = []
    with open('./data/basis.txt') as p:
        for test_line in p:
            tests.append(feature.convert_sentence(test_line))

    max_width = 35  # TODO: 自動で取れるように
    tests = [(test + [0] * (max_width - len(test)))[:max_width] for test in tests]
    svc = joblib.load('./model/random-forest.pkl')
    prediction = svc.predict(tests)

    with open('./data/output', mode='w') as p:
        p.write("\n".join(['1' if result == '+1' else '-1' for result in prediction.tolist()]))


if __name__ == '__main__':
    main()
