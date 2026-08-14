import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

class FireflyOptimizer:
    def __init__(self, population_size=5, generations=3,
                 alpha=0.3, beta_zero=1.0, gamma=0.001):
        self.population_size = population_size
        self.generations = generations
        self.alpha = alpha
        self.beta_zero = beta_zero
        self.gamma = gamma
        self.rng = np.random.default_rng(42)

    def create_firefly(self):
        return np.array([
            self.rng.integers(20, 101),
            self.rng.integers(2, 16)
        ], dtype=float)

    def evaluate(self, firefly, X_train, y_train, X_val, y_val):
        trees = int(np.clip(firefly[0], 20, 100))
        depth = int(np.clip(firefly[1], 2, 15))
        model = RandomForestRegressor(
            n_estimators=trees, max_depth=depth,
            random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        return mean_squared_error(y_val, model.predict(X_val))

    def optimize(self, X_train, y_train, X_val, y_val):
        fireflies = [self.create_firefly() for _ in range(self.population_size)]
        scores = [
            self.evaluate(f, X_train, y_train, X_val, y_val)
            for f in fireflies
        ]

        for _ in range(self.generations):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if scores[j] < scores[i]:
                        distance = np.linalg.norm(fireflies[i] - fireflies[j])
                        beta = self.beta_zero * np.exp(-self.gamma * distance ** 2)
                        move = self.alpha * (self.rng.random(2) - 0.5)
                        candidate = (
                            fireflies[i]
                            + beta * (fireflies[j] - fireflies[i])
                            + move
                        )
                        candidate[0] = np.clip(candidate[0], 20, 100)
                        candidate[1] = np.clip(candidate[1], 2, 15)
                        score = self.evaluate(
                            candidate, X_train, y_train, X_val, y_val
                        )
                        if score < scores[i]:
                            fireflies[i], scores[i] = candidate, score

        best = int(np.argmin(scores))
        return {
            "n_estimators": int(fireflies[best][0]),
            "max_depth": int(fireflies[best][1]),
            "best_mse": float(scores[best])
        }
