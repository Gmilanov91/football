"""
Neural Network Model - MLP and LSTM architectures for predictions
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from typing import Dict, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

np.random.seed(42)
tf.random.set_seed(42)


class FootballNeuralModel:
    """Neural Network model for football predictions"""
    
    def __init__(self, model_type: str = 'mlp', input_dim: int = 50):
        self.model_type = model_type
        self.input_dim = input_dim
        self.model = None
        self.history = None
        self.is_trained = False
        
        self.config = {
            'mlp_layers': [256, 128, 64, 32],
            'lstm_units': [128, 64],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
        }
    
    def build_mlp_model(self) -> keras.Model:
        """Build MLP model"""
        logger.info("Building MLP model")
        
        # Use unique model name with timestamp to avoid conflicts
        import time
        model_id = int(time.time() * 1000) % 1000000
        
        inputs = layers.Input(shape=(self.input_dim,), name=f'match_features_{model_id}')
        
        x = inputs
        for i, units in enumerate(self.config['mlp_layers']):
            x = layers.Dense(
                units, 
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(0.001),
                name=f'dense_{i+1}_{model_id}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_{i+1}_{model_id}')(x)
            x = layers.Dropout(self.config['dropout_rate'], name=f'dropout_{i+1}_{model_id}')(x)
        
        result_output = layers.Dense(3, activation='softmax', name=f'match_result_{model_id}')(x)
        
        home_goals_branch = layers.Dense(32, activation='relu', name=f'home_goals_branch_{model_id}')(x)
        home_goals_output = layers.Dense(1, activation='relu', name=f'home_goals_{model_id}')(home_goals_branch)
        
        away_goals_branch = layers.Dense(32, activation='relu', name=f'away_goals_branch_{model_id}')(x)
        away_goals_output = layers.Dense(1, activation='relu', name=f'away_goals_{model_id}')(away_goals_branch)
        
        over_2_5_output = layers.Dense(1, activation='sigmoid', name=f'over_2_5_{model_id}')(x)
        btts_output = layers.Dense(1, activation='sigmoid', name=f'btts_{model_id}')(x)
        
        model = keras.Model(
            inputs=inputs,
            outputs={
                'match_result': result_output,
                'home_goals': home_goals_output,
                'away_goals': away_goals_output,
                'over_2_5': over_2_5_output,
                'btts': btts_output
            },
            name=f'football_mlp_predictor_{model_id}'
        )
        
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss={
                'match_result': 'categorical_crossentropy',
                'home_goals': 'mse',
                'away_goals': 'mse',
                'over_2_5': 'binary_crossentropy',
                'btts': 'binary_crossentropy'
            },
            loss_weights={
                'match_result': 1.0,
                'home_goals': 0.5,
                'away_goals': 0.5,
                'over_2_5': 0.3,
                'btts': 0.3
            },
            metrics={
                'match_result': 'accuracy',
                'home_goals': 'mae',
                'away_goals': 'mae',
                'over_2_5': 'accuracy',
                'btts': 'accuracy'
            }
        )
        
        return model
    
    def build_model(self) -> keras.Model:
        """Build model based on type"""
        if self.model is not None:
            return self.model
        
        if self.model_type == 'lstm':
            # LSTM implementation can be added if needed
            logger.warning("LSTM not implemented, using MLP")
            self.model = self.build_mlp_model()
        else:
            self.model = self.build_mlp_model()
        
        return self.model
    
    def predict_single(self, features: np.ndarray) -> Dict:
        """Predict for single match"""
        if not self.is_trained or self.model is None:
            # Return baseline prediction if model not trained
            return self._baseline_prediction(features)
        
        # Ensure features is properly shaped
        features = np.asarray(features)
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        elif len(features.shape) > 2:
            features = features.flatten().reshape(1, -1)
        
        raw_predictions = self.model.predict(features, verbose=0)
        
        result_probs = raw_predictions['match_result'][0]
        
        return {
            'match_result': {
                'home_win_probability': float(result_probs[0]),
                'draw_probability': float(result_probs[1]),
                'away_win_probability': float(result_probs[2]),
                'predicted_outcome': ['Home Win', 'Draw', 'Away Win'][np.argmax(result_probs)],
                'confidence': float(np.max(result_probs))
            },
            'expected_goals': {
                'home': float(raw_predictions['home_goals'][0][0]),
                'away': float(raw_predictions['away_goals'][0][0]),
                'total': float(raw_predictions['home_goals'][0][0] + raw_predictions['away_goals'][0][0])
            },
            'betting_markets': {
                'over_2_5_probability': float(raw_predictions['over_2_5'][0][0]),
                'under_2_5_probability': float(1 - raw_predictions['over_2_5'][0][0]),
                'btts_probability': float(raw_predictions['btts'][0][0]),
                'btts_no_probability': float(1 - raw_predictions['btts'][0][0])
            }
        }
    
    def _baseline_prediction(self, features: np.ndarray) -> Dict:
        """Baseline prediction when model not trained"""
        # Ensure features is 1D array
        if len(features.shape) > 1:
            features = features.flatten()
        
        # Simple heuristic based on features
        home_strength = float(features[0]) if len(features) > 0 else 0.4
        away_strength = float(features[8]) if len(features) > 8 else 0.3
        
        total = home_strength + away_strength + 0.25
        
        return {
            'match_result': {
                'home_win_probability': home_strength / total if total > 0 else 0.45,
                'draw_probability': 0.25 / total if total > 0 else 0.30,
                'away_win_probability': away_strength / total if total > 0 else 0.25,
                'predicted_outcome': 'Home Win' if home_strength > away_strength else 'Away Win',
                'confidence': 0.5
            },
            'expected_goals': {
                'home': max(0, home_strength * 2.5),
                'away': max(0, away_strength * 2.0),
                'total': max(0, (home_strength * 2.5) + (away_strength * 2.0))
            },
            'betting_markets': {
                'over_2_5_probability': 0.5,
                'under_2_5_probability': 0.5,
                'btts_probability': 0.5,
                'btts_no_probability': 0.5
            }
        }

