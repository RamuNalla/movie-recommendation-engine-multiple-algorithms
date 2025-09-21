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

        self.global_mean = np.mean(user_item_matrix[user_item_matrix != 0])         # calculate means for bias terms

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
        self.item_factors = self.svd.components_.T
