import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def linear_regression(X,y):
    model = LinearRegression()
    model.fit(X,y)
    R2 = model.score(X,y)
    return np.hstack((np.array(model.intercept_), model.coef_)), R2


class ClusterwiseLR():
    def __init__(self, k, gamma, max_iter=200):
        self.k = k
        self.gamma = gamma
        self.labels = None
        self.models = None
        self.max_iter = max_iter
        self.clusters = [i for i in range(self.k)]
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        it = 0
        self.labels = np.random.randint(0, self.k, X.shape[0])
        self.models = np.zeros((self.k, X.shape[1] + 1))
        while it < self.max_iter:
            # Controlla se ci sono cluster vuoti e ripartisci campioni
            for cluster in range(self.k):
                if np.sum(self.labels == cluster) == 0:
                    print(f"Cluster {cluster} is empty.")
                    continue

            print(f"Iteration {it}: {np.unique(self.labels, return_counts=True)}")

            # Esegui la regressione per ogni cluster
            for cluster in range(self.k):
                if np.sum(self.labels == cluster) > 0:
                    self.models[cluster, :] = \
                    linear_regression(X[self.labels == cluster, :], y[self.labels == cluster])[0]

            # Riassegna le etichette basate sull'errore minimo
            for obs in range(X.shape[0]):
                label = 0
                error = np.inf
                for cluster in range(self.k):
                    error_term = (np.dot(np.hstack((1, X[obs, :])), self.models[cluster, :]) - y[obs])
                    reg_term = np.linalg.norm(
                        X[obs, :] - (1 / max(1, np.sum(self.labels == cluster))) * np.sum(X[self.labels == cluster, :]))
                    total_error = error_term ** 2 + self.gamma * (reg_term ** 2)
                    if total_error < error:
                        error = total_error
                        label = cluster
                self.labels[obs] = label

            it += 1

    def predict(self, x_new, test_labels_prediction="k_plane"):
        bel_cluster = None
        if test_labels_prediction == "k_plane":
            dist = np.Inf
            for cluster in self.clusters:
                dist_ = np.linalg.norm(
                    x_new - (1 / len(self.labels == cluster)) * np.sum(X[self.labels == cluster, :])) ** 2
                if dist_ < dist:
                    bel_cluster = cluster
                    dist = dist_
        elif test_labels_prediction == "logistic_regression":
            x_new_pred = x_new.reshape(1, -1)
            lr = LogisticRegression().fit(X, self.labels)
            bel_cluster = lr.predict(x_new_pred)
        elif test_labels_prediction == "Random forest":
            x_new_pred = x_new.reshape(1, -1)
            clf = RandomForestClassifier(random_state=0).fit(X, self.labels)
            bel_cluster = clf.predict(x_new_pred)

        y_predicted = np.dot(np.hstack((1, x_new)), np.squeeze(self.models[bel_cluster, :]))
        return y_predicted, bel_cluster

    def performance_dataframe(self, colnames):
        non_empty_clusters = [c for c in range(self.k) if np.sum(self.labels == c) >= 1]
        df_dict = {f"Cluster {i}": self.models[i, :].tolist() + [
            linear_regression(self.X[self.labels == i, :], self.y[self.labels == i])[1]] for i in non_empty_clusters}
        df = pd.DataFrame.from_dict(df_dict, orient='index', columns=colnames)
        return df



