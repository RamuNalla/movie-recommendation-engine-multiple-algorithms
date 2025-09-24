import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class NeuralCollaborativeFIltering:         # Neural Collaborative filtering (combines matrix factorization and Multi-layer perceptron)

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 50, 
                 hidden_units: List[int] = [128, 64, 32], dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None

    def build_model(self):          # Build NCF model

        user_input = layers.Input(shape=(), name='user_id')     # Input layers
        item_input = layers.Input(shape=(), name='item_id')
        
        # Embedding layers for Matrix Factorization path
        user_embedding_mf = layers.Embedding(
            self.n_users, self.embedding_dim, name='user_embedding_mf'
        )(user_input)
        item_embedding_mf = layers.Embedding(
            self.n_items, self.embedding_dim, name='item_embedding_mf'
        )(item_input)
        
        # Embedding layers for MLP path (larger dimension)
        user_embedding_mlp = layers.Embedding(
            self.n_users, self.embedding_dim, name='user_embedding_mlp'
        )(user_input)
        item_embedding_mlp = layers.Embedding(
            self.n_items, self.embedding_dim, name='item_embedding_mlp'
        )(item_input)
        
        # Flatten embeddings
        user_vec_mf = layers.Flatten()(user_embedding_mf)
        item_vec_mf = layers.Flatten()(item_embedding_mf)
        user_vec_mlp = layers.Flatten()(user_embedding_mlp)
        item_vec_mlp = layers.Flatten()(item_embedding_mlp)
        
        mf_output = layers.Multiply()([user_vec_mf, item_vec_mf])       # Matrix Factorization path - element-wise product
        
        mlp_input = layers.Concatenate()([user_vec_mlp, item_vec_mlp])  # MLP path - concatenate user and item embeddings
        
        # MLP layers
        mlp_output = mlp_input
        for units in self.hidden_units:
            mlp_output = layers.Dense(units, activation='relu')(mlp_output)
            mlp_output = layers.Dropout(self.dropout_rate)(mlp_output)
        
        combined = layers.Concatenate()([mf_output, mlp_output])                # Combine MF and MLP outputs
        
        output = layers.Dense(1, activation='linear', name='rating')(combined)  # Final prediction layer
        
        self.model = Model(inputs=[user_input, item_input], outputs=output)     # Create model
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model

    def fit(self, train_df: pd.DataFrame, validation_df: pd.DataFrame = None,       # Train the NCF model
            epochs: int = 50, batch_size: int = 256, verbose: int = 1):
        
        if self.model is None:
            self.build_model()
        
        # Prepare training data
        X_train = [train_df['user_idx'].values, train_df['item_idx'].values]
        y_train = train_df['rating'].values
        
        # Prepare validation data
        validation_data = None
        if validation_df is not None:
            X_val = [validation_df['user_idx'].values, validation_df['item_idx'].values]
            y_val = validation_df['rating'].values
            validation_data = (X_val, y_val)
        
        # Train model
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

    def predict(self, user_idx: int, item_idx: int) -> float:       # Predict rating for user-item pair
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        prediction = self.model.predict([[user_idx], [item_idx]], verbose=0)[0, 0]
        return np.clip(prediction, 1, 5)
    
    def recommend_items(self, user_idx: int, user_item_matrix: np.ndarray,      # Recommend top-N items for a user
                       n_recommendations: int = 10) -> List[Tuple[int, float]]:

        unrated_items = np.where(user_item_matrix[user_idx, :] == 0)[0]
        
        if len(unrated_items) == 0:
            return []
        
        # Batch prediction for efficiency
        user_indices = np.full(len(unrated_items), user_idx)
        predictions = self.model.predict([user_indices, unrated_items], verbose=0)
        
        # Create recommendation list
        recommendations = [(item_idx, pred[0]) for item_idx, pred in zip(unrated_items, predictions)]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]
    

class AutoEncoder:      # Autoencoder for collaborative filtering

    def __init__(self, input_dim: int, encoding_dims: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.2, learning_rate: float = 0.001):
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.encoder = None

    def build_model(self):                  # Build Autoencoder model
        
        input_layer = layers.Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = input_layer
        for dim in self.encoding_dims:
            encoded = layers.Dense(dim, activation='relu')(encoded)
            encoded = layers.Dropout(self.dropout_rate)(encoded)
        
        # Decoder (symmetric)
        decoded = encoded
        for dim in reversed(self.encoding_dims[:-1]):
            decoded = layers.Dense(dim, activation='relu')(decoded)
            decoded = layers.Dropout(self.dropout_rate)(decoded)
        
        # Output layer
        output = layers.Dense(self.input_dim, activation='linear')(decoded)
        
        # Create models
        self.model = Model(input_layer, output)
        self.encoder = Model(input_layer, encoded)
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return self.model
    
    def fit(self, user_item_matrix: np.ndarray, validation_split: float = 0.1,      # Train the autoencoder
            epochs: int = 100, batch_size: int = 32, verbose: int = 1):
        
        if self.model is None:
            self.build_model()
        
        # Use only users with at least one rating
        active_users = np.sum(user_item_matrix, axis=1) > 0
        X_train = user_item_matrix[active_users]
        
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ]
        
        self.history = self.model.fit(
            X_train, X_train,  # AutoEncoder reconstructs input
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
    
    def predict(self, user_idx: int, item_idx: int, user_item_matrix: np.ndarray) -> float:     # Predict rating for user-item pair
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        user_vector = user_item_matrix[user_idx:user_idx+1]
        reconstructed = self.model.predict(user_vector, verbose=0)
        prediction = reconstructed[0, item_idx]
        
        return np.clip(prediction, 1, 5)

    def get_user_embeddings(self, user_item_matrix: np.ndarray) -> np.ndarray:      # Get user embeddings from encoder
        
        if self.encoder is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        return self.encoder.predict(user_item_matrix, verbose=0)
    
class WideAndDeep:          # Wide and Deep Learning for Recommender Systems

    def __init__(self, n_users: int, n_items: int, user_features: np.ndarray = None,
                 item_features: np.ndarray = None, embedding_dim: int = 32,
                 deep_hidden_units: List[int] = [128, 64], dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        self.n_users = n_users
        self.n_items = n_items
        self.user_features = user_features
        self.item_features = item_features
        self.embedding_dim = embedding_dim
        self.deep_hidden_units = deep_hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None

    def build_model(self):                  # Build Wide & Deep model
        
        # Input layers
        user_input = layers.Input(shape=(), name='user_id')
        item_input = layers.Input(shape=(), name='item_id')
        
        inputs = [user_input, item_input]
        
        # User and item embeddings for deep part
        user_embedding = layers.Embedding(
            self.n_users, self.embedding_dim, name='user_embedding'
        )(user_input)
        item_embedding = layers.Embedding(
            self.n_items, self.embedding_dim, name='item_embedding'
        )(item_input)
        
        user_vec = layers.Flatten()(user_embedding)
        item_vec = layers.Flatten()(item_embedding)
        
        # Wide part - direct connections and feature crosses
        wide_inputs = []
        
        # Add user and item IDs to wide part (one-hot encoded)
        user_onehot = layers.Lambda(
            lambda x: tf.one_hot(tf.cast(x, tf.int32), self.n_users)
        )(user_input)
        item_onehot = layers.Lambda(
            lambda x: tf.one_hot(tf.cast(x, tf.int32), self.n_items)
        )(item_input)
        
        wide_inputs.extend([user_onehot, item_onehot])
        
        # Add additional features if provided
        if self.user_features is not None:
            user_feat_input = layers.Input(shape=(self.user_features.shape[1],), name='user_features')
            inputs.append(user_feat_input)
            wide_inputs.append(user_feat_input)
        
        if self.item_features is not None:
            item_feat_input = layers.Input(shape=(self.item_features.shape[1],), name='item_features')
            inputs.append(item_feat_input)
            wide_inputs.append(item_feat_input)
        
        # Deep part - neural network
        deep_input = layers.Concatenate()([user_vec, item_vec])
        
        # Add feature inputs to deep part
        deep_additional = []
        if self.user_features is not None:
            deep_additional.append(user_feat_input)
        if self.item_features is not None:
            deep_additional.append(item_feat_input)
        
        if deep_additional:
            deep_input = layers.Concatenate()([deep_input] + deep_additional)
        
        # Deep layers
        deep_output = deep_input
        for units in self.deep_hidden_units:
            deep_output = layers.Dense(units, activation='relu')(deep_output)
            deep_output = layers.Dropout(self.dropout_rate)(deep_output)
        
        # Combine wide and deep
        if wide_inputs:
            wide_output = layers.Concatenate()(wide_inputs)
            combined = layers.Concatenate()([wide_output, deep_output])
        else:
            combined = deep_output
        
        # Final output
        output = layers.Dense(1, activation='linear', name='rating')(combined)
        
        # Create and compile model
        self.model = Model(inputs=inputs, outputs=output)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def fit(self, train_df: pd.DataFrame, validation_df: pd.DataFrame = None,           # Train Wide & Deep model
            epochs: int = 50, batch_size: int = 256, verbose: int = 1):
        
        if self.model is None:
            self.build_model()
        
        # Prepare training data
        X_train = [train_df['user_idx'].values, train_df['item_idx'].values]
        
        if self.user_features is not None:
            X_train.append(self.user_features[train_df['user_idx'].values])
        if self.item_features is not None:
            X_train.append(self.item_features[train_df['item_idx'].values])
        
        y_train = train_df['rating'].values
        
        # Prepare validation data
        validation_data = None
        if validation_df is not None:
            X_val = [validation_df['user_idx'].values, validation_df['item_idx'].values]
            
            if self.user_features is not None:
                X_val.append(self.user_features[validation_df['user_idx'].values])
            if self.item_features is not None:
                X_val.append(self.item_features[validation_df['item_idx'].values])
            
            y_val = validation_df['rating'].values
            validation_data = (X_val, y_val)
        
        # Train model
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

    def predict(self, user_idx: int, item_idx: int) -> float:           # Predict rating for user-item pair
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        X_pred = [[user_idx], [item_idx]]
        
        if self.user_features is not None:
            X_pred.append(self.user_features[user_idx:user_idx+1])
        if self.item_features is not None:
            X_pred.append(self.item_features[item_idx:item_idx+1])
        
        prediction = self.model.predict(X_pred, verbose=0)[0, 0]
        return np.clip(prediction, 1, 5)

    



    


