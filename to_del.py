def fit(self, X, y):
    self.X = X
    self.y = y
    it = 0
    self.labels = np.random.randint(0, self.k, X.shape[0])
    self.models = np.zeros((self.k, X.shape[1] + 1))
    while it < self.max_iter:
        print(np.unique_counts(self.labels))
        for cluster in self.clusters:
            self.models[cluster, :] = linear_regression(X[self.labels == cluster, :], y[self.labels == cluster])[0]
        for obs in range(X.shape[0]):
            label = 0
            error = np.inf
            for cluster in self.clusters:
                error_term = (np.dot(np.hstack((1, X[obs, :])), self.models[cluster, :]) - y[obs])
                reg_term = np.linalg.norm(
                    X[obs, :] - (1 / len(self.labels == cluster)) * np.sum(X[self.labels == cluster, :]))
                total_error = error_term ** 2 + self.gamma * (reg_term ** 2)
                if total_error < error:
                    error = total_error
                    label = cluster
            self.labels[obs] = label

        it = it + 1