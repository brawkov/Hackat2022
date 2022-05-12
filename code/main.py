import pickle
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import signal_io


def set_cluster(df):
    cluster_nan = df[df["cluster"] == -1]
    df_predict = cluster_nan.drop(['cluster', 'p0', 'p1', 'p2', 'p3'], axis=1)

    filename = 'cluster_model.sav'
    clf = pickle.load(open(filename, 'rb'))

    cluster_result = clf.predict(df_predict)
    print(type(cluster_result))
    cluster_nan['cluster'] = clf.predict(df_predict)
    print(cluster_nan['cluster'])


if __name__ == "__main__":
    df = signal_io.read_signals('../data/signals.csv')
    set_cluster(df)

