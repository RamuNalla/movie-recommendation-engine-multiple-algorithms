import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics import mean_squared_error
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class SVDRecommender:       # SVD for collaborative filtering (uses TruncatedSVD for dimensionality reduction)

    def __init__(self, n_factors = 100, random_state = 42):
        self.n_factors = n_factors
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_factors, random_state=random_state)
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None
        self.user_means = None
        self.item_means = None

    def fit(self, user_item_matrix: np.ndarray):        # fit SVD model to User-Item matrix

        self.global_mean = np.mean(user_item_matrix[user_item_matrix != 0])         # calculate means for bias terms (average rating of all non-zero ratings)

        self.user_means = np.zeros(user_item_matrix.shape[0])           # user biases
        self.item_means = np.zeros(user_item_matrix.shape[1])           # item biases

        for i in range(user_item_matrix.shape[0]):
            user_ratings = user_item_matrix[i][user_item_matrix[i] != 0]
            if len(user_ratings) > 0:
                self.user_means[i] = np.mean(user_ratings) - self.global_mean

        for j in range(user_item_matrix.shape[1]):
            item_ratings = user_item_matrix[:, j][user_item_matrix[:, j] != 0]
            if len(item_ratings) > 0:
                self.item_means[j] = np.mean(item_ratings) - self.global_mean

        self.user_factors = self.svd.fit_transform(user_item_matrix)            # apply SVD
        self.item_factors = self.svd.components_.T                              # The product of these two would approximate the original matrix

    
    def predict(self, user_idx: int, item_idx: int) -> float:               # predict rating for user-item pair

        if self.user_factors is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prediction = global_mean + user_bias + item_bias + user_factors @ item_factors
        prediction = (self.global_mean + 
                     self.user_means[user_idx] + 
                     self.item_means[item_idx] +
                     np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
        
        return np.clip(prediction, 1, 5)
    
    def recommend_items(self, user_idx: int, user_item_matrix: np.ndarray, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        # Recommend top-N items for a user, returns a list of (item_idx, predicted_rating) tuples

        unrated_items = np.where(user_item_matrix[user_idx, :] == 0)[0]     # get unrated items

        predictions = []
        for item_idx in unrated_items:
            pred = self.predict(user_idx, item_idx)
            predictions.append((item_idx, pred))

        predictions.sort(key=lambda x: x[1], reverse=True)      # sort by predicted rating

        return predictions[:n_recommendations]
    

class NMFRecommender:           # Non-negative Matrix Factorization for Recommendation (good for interpretable latent factors)

    def __init__(self, n_factors=50, max_iter=200, random_state=42):
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.random_state = random_state
        self.nmf = NMF(n_components=n_factors, max_iter=max_iter, random_state=random_state, init='random')
        self.user_factors = None
        self.item_factors = None

    def fit(self, user_item_matrix: np.ndarray):

        shifted_matrix = user_item_matrix.copy()            # Ensure non-negative matrix (shift ratings from 1-5 to 0-4)
        shifted_matrix[shifted_matrix != 0] -= 1
        
        self.user_factors = self.nmf.fit_transform(shifted_matrix)      # apply NMF
        self.item_factors = self.nmf.components_.T

    def predict(self, user_idx: int, item_idx: int) -> float:       # predict rating for user-item pair

        if self.user_factors is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx]) + 1       # Prediction = user_factors @ item_factors + 1 (shift back)
        return np.clip(prediction, 1, 5)
    
    def recommend_items(self, user_idx: int, user_item_matrix: np.ndarray, 
                       n_recommendations: int = 10) -> List[Tuple[int, float]]:         # Recommend top-N items for a user
        
        unrated_items = np.where(user_item_matrix[user_idx, :] == 0)[0]
        
        predictions = []
        for item_idx in unrated_items:
            pred = self.predict(user_idx, item_idx)
            predictions.append((item_idx, pred))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
class ALSRecommender:       # Alternative Least Squares (ALS) Matrix Factorization. Popular in industry

    def __init__(self, n_factors=100, reg_param=0.1, max_iter=10, random_state=42):
        self.n_factors = n_factors
        self.reg_param = reg_param
        self.max_iter = max_iter
        self.random_state = random_state
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None

    def fit(self, user_item_matrix: np.ndarray):        # fit the ALS model using alternating least squares

        np.random.seed(self.random_state)

        n_users, n_items = user_item_matrix.shape
        self.global_mean = np.mean(user_item_matrix[user_item_matrix != 0])

        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))     # Initialize factor matrices randomly
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))

        mask = user_item_matrix != 0            # create mask for observed ratings

        for iteration in range(self.max_iter):

            for u in range(n_users):            # update user factors
                rated_items = mask[u, :]
                if np.sum(rated_items) > 0:
                    X_u = self.item_factors[rated_items]
                    y_u = user_item_matrix[u, rated_items]

                    A = X_u.T @ X_u + self.reg_param * np.eye(self.n_factors)       # Solve: (X^T X + λI) θ = X^T y
                    b = X_u.T @ y_u
                    self.user_factors[u] = np.linalg.solve(A, b)

            for i in range(n_items):            # update item factors
                rating_users = mask[:, i]
                if np.sum(rating_users) > 0:
                    X_i = self.user_factors[rating_users]
                    y_i = user_item_matrix[rating_users, i]
                    
                    A = X_i.T @ X_i + self.reg_param * np.eye(self.n_factors)
                    b = X_i.T @ y_i
                    self.item_factors[i] = np.linalg.solve(A, b)
            
            if iteration % 5 == 0:              # calculate training error
                predictions = self.user_factors @ self.item_factors.T
                mse = mean_squared_error(
                    user_item_matrix[mask], 
                    predictions[mask]
                )
                print(f"Iteration {iteration}, MSE: {mse:.4f}")

    def predict(self, user_idx: int, item_idx: int) -> float:       # Predict rating for user-item pair

        if self.user_factors is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        return np.clip(prediction, 1, 5)
    
    def recommend_items(self, user_idx: int, user_item_matrix: np.ndarray, 
                       n_recommendations: int = 10) -> List[Tuple[int, float]]:     # recommend top-N items for a user

        unrated_items = np.where(user_item_matrix[user_idx, :] == 0)[0]
        
        predictions = []
        for item_idx in unrated_items:
            pred = self.predict(user_idx, item_idx)
            predictions.append((item_idx, pred))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]


class BiasedMF:         # Matrix factorization with User and Item biases (bias model from Netflix prize)

    def __init__(self, n_factors=100, learning_rate=0.01, reg_param=0.01, 
                 n_epochs=100, random_state=42):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg_param = reg_param
        self.n_epochs = n_epochs
        self.random_state = random_state
        
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None

    def fit(self, ratings_df: pd.DataFrame):        # fit model using stochastic gradient descent model, ratings_df: DataFrame with columns ['user_idx', 'item_idx', 'rating']

        np.random.seed(self.random_state)
        
        n_users = ratings_df['user_idx'].max() + 1
        n_items = ratings_df['item_idx'].max() + 1

        self.global_mean = ratings_df['rating'].mean()          # Initialize parameters
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)

        for epoch in range(self.n_epochs):

            shuffled_df = ratings_df.sample(frac=1).reset_index(drop=True)  # shuffle training data

            epoch_loss = 0
            for _, row in shuffled_df.iterrows():
                user_idx = int(row['user_idx'])
                item_idx = int(row['item_idx'])
                rating = row['rating']
            
                pred = (self.global_mean +              # prediction
                        self.user_biases[user_idx] + 
                        self.item_biases[item_idx] +
                        np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
                
                error = rating - pred               # error
                epoch_loss += error ** 2

                # Update biases
                self.user_biases[user_idx] += self.learning_rate * (error - self.reg_param * self.user_biases[user_idx])
                self.item_biases[item_idx] += self.learning_rate * (error - self.reg_param * self.item_biases[item_idx])

                # Update factors
                user_factors_old = self.user_factors[user_idx].copy()
                self.user_factors[user_idx] += self.learning_rate * (error * self.item_factors[item_idx] - 
                                                                    self.reg_param * self.user_factors[user_idx])
                self.item_factors[item_idx] += self.learning_rate * (error * user_factors_old - 
                                                                    self.reg_param * self.item_factors[item_idx])
            
            if epoch % 10 == 0:
                    rmse = np.sqrt(epoch_loss / len(shuffled_df))
                    print(f"Epoch {epoch}, RMSE: {rmse:.4f}")

    def predict(self, user_idx: int, item_idx: int) -> float:       # Predict rating for user-item pair

        if self.user_factors is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        prediction = (self.global_mean + 
                     self.user_biases[user_idx] + 
                     self.item_biases[item_idx] +
                     np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
        
        return np.clip(prediction, 1, 5)
    
    def recommend_items(self, user_idx: int, user_item_matrix: np.ndarray,      # Recommend top-N items for a user
                       n_recommendations: int = 10) -> List[Tuple[int, float]]:

        unrated_items = np.where(user_item_matrix[user_idx, :] == 0)[0]
        
        predictions = []
        for item_idx in unrated_items:
            pred = self.predict(user_idx, item_idx)
            predictions.append((item_idx, pred))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

    

                    
                




            


    

    






    

    







    
