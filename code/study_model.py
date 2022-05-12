import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import signal_io


def study_model():
    df = signal_io.read_signals('../data/signals.csv')

    # выбора для обучения в которой заполненые значения
    cluster = df[df["cluster"] != -1]
    df_x = cluster.drop(['cluster', 'p0', 'p1', 'p2', 'p3'], axis=1)
    df_y = cluster["cluster"]
    # print(df_x)
    # print(df_y)
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.1)

    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)

    # обучаем
    clf = RandomForestClassifier(n_estimators=150, random_state=50, criterion='entropy')
    clf.fit(x_train, y_train)
    # смотрим точность
    print(clf.score(x_test, y_test))

    # Записываем в файл обученую модель
    filename = 'cluster_model.sav'
    pickle.dump(clf, open(filename, 'wb'))


if __name__ == "__main__":
    study_model()
