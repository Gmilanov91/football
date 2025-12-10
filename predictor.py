"""
Main Prediction Engine - Integrates all modules
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging
from datetime import datetime

from data_scraper import FootballDataScraper
from data_preprocessor import FootballDataPreprocessor
from feature_engineer import FootballFeatureEngineer
from neural_model import FootballNeuralModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FootballPredictor:
    """Complete football prediction system"""
    
    def __init__(self, model_type: str = 'mlp'):
        logger.info("Initializing Football Predictor")
        
        self.scraper = FootballDataScraper(cache_duration_hours=6)
        self.preprocessor = FootballDataPreprocessor()
        self.feature_engineer = FootballFeatureEngineer()
        
        self.model_type = model_type
        self.model = None
    
    def predict_match(
        self, 
        home_team: str, 
        away_team: str,
        league: str = "Premier League",
        return_details: bool = True
    ) -> Dict:
        """Predict match outcome"""
        logger.info(f"Predicting: {home_team} vs {away_team}")
        
        try:
            # Fetch data
            raw_data = self.scraper.fetch_match_data(home_team, away_team, league)
            
            # Preprocess
            preprocessed_data = self.preprocessor.preprocess_match_data(raw_data)
            
            # Engineer features
            features_df = self.feature_engineer.engineer_features(preprocessed_data)
            
            # Normalize
            normalized_features = self.preprocessor.normalize_features(features_df, fit=False)
            
            # Predict
            if self.model is None:
                input_dim = normalized_features.shape[1]
                self.model = FootballNeuralModel(model_type=self.model_type, input_dim=input_dim)
                # Build model only once
                if self.model.model is None:
                    self.model.build_model()
                self.model.is_trained = False  # Model not trained, will use baseline
            
            features_array = normalized_features.values
            # Ensure it's a numpy array and flatten if needed
            if isinstance(features_array, pd.DataFrame):
                features_array = features_array.values
            if len(features_array.shape) > 1 and features_array.shape[0] == 1:
                features_array = features_array[0]
            
            prediction = self.model.predict_single(features_array)
            
            # Add details
            if return_details:
                prediction = self._add_prediction_details(prediction, preprocessed_data, features_df)
            
            prediction['metadata'] = {
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'prediction_time': datetime.now().isoformat(),
                'model_type': self.model_type
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'error': str(e),
                'home_team': home_team,
                'away_team': away_team
            }
    
    def _add_prediction_details(self, prediction: Dict, preprocessed_data: Dict, features_df: pd.DataFrame) -> Dict:
        """Add detailed analysis"""
        key_factors = []
        
        home_form = preprocessed_data.get('home_form', {}).get('form_score', 0.5)
        away_form = preprocessed_data.get('away_form', {}).get('form_score', 0.5)
        
        if home_form > 0.7:
            key_factors.append("Home team in excellent form")
        elif home_form < 0.3:
            key_factors.append("Home team struggling with poor form")
        
        if away_form > 0.7:
            key_factors.append("Away team in excellent form")
        elif away_form < 0.3:
            key_factors.append("Away team struggling with poor form")
        
        h2h = preprocessed_data.get('h2h_stats', {})
        if h2h.get('team1_win_rate', 0) > 0.6:
            key_factors.append("Home team dominates head-to-head history")
        elif h2h.get('team2_win_rate', 0) > 0.6:
            key_factors.append("Away team dominates head-to-head history")
        
        home_avail = preprocessed_data.get('home_availability', {})
        away_avail = preprocessed_data.get('away_availability', {})
        
        if home_avail.get('key_players_missing', 0) > 2:
            key_factors.append("Home team missing key players")
        if away_avail.get('key_players_missing', 0) > 2:
            key_factors.append("Away team missing key players")
        
        prediction['analysis'] = {
            'key_factors': key_factors,
            'home_form': {
                'score': home_form,
                'rating': self._get_form_rating(home_form)
            },
            'away_form': {
                'score': away_form,
                'rating': self._get_form_rating(away_form)
            },
            'head_to_head': {
                'home_wins': h2h.get('team1_wins', 0),
                'draws': h2h.get('draws', 0),
                'away_wins': h2h.get('team2_wins', 0),
                'avg_goals': h2h.get('avg_goals', 2.5)
            }
        }
        
        prediction['betting_insights'] = self._generate_betting_insights(prediction)
        
        return prediction
    
    def _get_form_rating(self, form_score: float) -> str:
        """Get form rating"""
        if form_score >= 0.7:
            return "Excellent"
        elif form_score >= 0.5:
            return "Good"
        elif form_score >= 0.3:
            return "Average"
        else:
            return "Poor"
    
    def _generate_betting_insights(self, prediction: Dict) -> Dict:
        """Generate betting insights"""
        insights = {}
        
        result = prediction['match_result']
        if result['confidence'] > 0.7:
            insights['match_result'] = {
                'recommendation': f"Strong {result['predicted_outcome']}",
                'confidence': 'High'
            }
        elif result['confidence'] > 0.5:
            insights['match_result'] = {
                'recommendation': f"Moderate {result['predicted_outcome']}",
                'confidence': 'Medium'
            }
        else:
            insights['match_result'] = {
                'recommendation': "Uncertain outcome",
                'confidence': 'Low'
            }
        
        betting = prediction['betting_markets']
        if betting['over_2_5_probability'] > 0.65:
            insights['goals'] = "Over 2.5 goals likely"
        elif betting['under_2_5_probability'] > 0.65:
            insights['goals'] = "Under 2.5 goals likely"
        else:
            insights['goals'] = "Goals market uncertain"
        
        if betting['btts_probability'] > 0.6:
            insights['btts'] = "Both teams likely to score"
        elif betting['btts_no_probability'] > 0.6:
            insights['btts'] = "Clean sheet likely"
        else:
            insights['btts'] = "BTTS market uncertain"
        
        return insights

