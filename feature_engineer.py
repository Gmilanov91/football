"""
Feature Engineering Module - Creates advanced ML features
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FootballFeatureEngineer:
    """Creates advanced features for match prediction"""
    
    def __init__(self):
        self.feature_names = []
    
    def engineer_features(self, preprocessed_data: Dict) -> pd.DataFrame:
        """Create comprehensive feature set"""
        logger.info("Engineering features")
        
        features_dict = {}
        
        features_dict.update(self._create_basic_features(preprocessed_data))
        features_dict.update(self._create_strength_features(preprocessed_data))
        features_dict.update(self._create_form_features(preprocessed_data))
        features_dict.update(self._create_h2h_features(preprocessed_data))
        features_dict.update(self._create_matchup_features(preprocessed_data))
        features_dict.update(self._create_availability_features(preprocessed_data))
        features_dict.update(self._create_home_advantage_features(preprocessed_data))
        features_dict.update(self._create_trend_features(preprocessed_data))
        
        features_df = pd.DataFrame([features_dict])
        self.feature_names = list(features_df.columns)
        
        return features_df
    
    def _create_basic_features(self, data: Dict) -> Dict:
        """Basic team statistics"""
        home = data['home_stats']
        away = data['away_stats']
        
        return {
            'home_win_rate': home.get('win_rate', 0),
            'home_draw_rate': home.get('draw_rate', 0),
            'home_avg_goals_scored': home.get('avg_goals_scored', 0),
            'home_avg_goals_conceded': home.get('avg_goals_conceded', 0),
            'home_points_per_game': home.get('points_per_game', 0),
            'home_clean_sheet_rate': home.get('clean_sheet_rate', 0),
            'home_scoring_rate': home.get('scoring_rate', 0),
            'away_win_rate': away.get('win_rate', 0),
            'away_draw_rate': away.get('draw_rate', 0),
            'away_avg_goals_scored': away.get('avg_goals_scored', 0),
            'away_avg_goals_conceded': away.get('avg_goals_conceded', 0),
            'away_points_per_game': away.get('points_per_game', 0),
            'away_clean_sheet_rate': away.get('clean_sheet_rate', 0),
            'away_scoring_rate': away.get('scoring_rate', 0),
            'home_home_win_rate': home.get('home_win_rate', 0),
            'home_home_ppg': home.get('home_points_per_game', 0),
            'away_away_win_rate': away.get('away_win_rate', 0),
            'away_away_ppg': away.get('away_points_per_game', 0),
        }
    
    def _create_strength_features(self, data: Dict) -> Dict:
        """Team strength comparisons"""
        home = data['home_stats']
        away = data['away_stats']
        
        return {
            'win_rate_diff': home.get('win_rate', 0) - away.get('win_rate', 0),
            'ppg_diff': home.get('points_per_game', 0) - away.get('points_per_game', 0),
            'attack_strength_diff': home.get('avg_goals_scored', 0) - away.get('avg_goals_conceded', 0),
            'defense_strength_diff': away.get('avg_goals_scored', 0) - home.get('avg_goals_conceded', 0),
            'goal_diff_comparison': home.get('goal_difference', 0) - away.get('goal_difference', 0),
            'strength_ratio': self._safe_divide(
                home.get('points_per_game', 1),
                away.get('points_per_game', 1)
            ),
            'home_advantage_strength': home.get('home_win_rate', 0) - away.get('away_win_rate', 0),
        }
    
    def _create_form_features(self, data: Dict) -> Dict:
        """Recent form features"""
        home_form = data.get('home_form', {})
        away_form = data.get('away_form', {})
        
        return {
            'home_form_score': home_form.get('form_score', 0.5),
            'away_form_score': away_form.get('form_score', 0.5),
            'form_diff': home_form.get('form_score', 0.5) - away_form.get('form_score', 0.5),
            'home_recent_goals_scored': home_form.get('avg_goals_scored', 0),
            'away_recent_goals_scored': away_form.get('avg_goals_scored', 0),
            'home_recent_goals_conceded': home_form.get('avg_goals_conceded', 0),
            'away_recent_goals_conceded': away_form.get('avg_goals_conceded', 0),
            'home_momentum': self._calculate_momentum(home_form),
            'away_momentum': self._calculate_momentum(away_form),
            'home_recent_win_rate': home_form.get('win_rate', 0),
            'away_recent_win_rate': away_form.get('win_rate', 0),
        }
    
    def _create_h2h_features(self, data: Dict) -> Dict:
        """Head-to-head features"""
        h2h = data.get('h2h_stats', {})
        
        return {
            'h2h_home_win_rate': h2h.get('team1_win_rate', 0.33),
            'h2h_draw_rate': h2h.get('draw_rate', 0.33),
            'h2h_away_win_rate': h2h.get('team2_win_rate', 0.33),
            'h2h_avg_goals': h2h.get('avg_goals', 2.5),
            'h2h_btts_rate': h2h.get('btts_rate', 0.5),
            'h2h_over_2_5_rate': h2h.get('over_2_5_rate', 0.5),
            'h2h_dominance': h2h.get('team1_win_rate', 0.33) - h2h.get('team2_win_rate', 0.33),
        }
    
    def _create_matchup_features(self, data: Dict) -> Dict:
        """Attack vs defense matchups"""
        home = data['home_stats']
        away = data['away_stats']
        
        expected_home_goals = max(0, (
            home.get('avg_goals_scored', 0) + 
            away.get('avg_goals_conceded', 0)
        ) / 2)
        
        expected_away_goals = max(0, (
            away.get('avg_goals_scored', 0) + 
            home.get('avg_goals_conceded', 0)
        ) / 2)
        
        total_goals = expected_home_goals + expected_away_goals
        
        return {
            'home_attack_vs_away_defense': (
                home.get('avg_goals_scored', 0) - away.get('avg_goals_conceded', 0)
            ),
            'away_attack_vs_home_defense': (
                away.get('avg_goals_scored', 0) - home.get('avg_goals_conceded', 0)
            ),
            'expected_home_goals': expected_home_goals,
            'expected_away_goals': expected_away_goals,
            'expected_total_goals': total_goals,
            'home_clean_sheet_prob': home.get('clean_sheet_rate', 0) * (1 - away.get('scoring_rate', 0.5)),
            'away_clean_sheet_prob': away.get('clean_sheet_rate', 0) * (1 - home.get('scoring_rate', 0.5)),
            'btts_probability': home.get('scoring_rate', 0.5) * away.get('scoring_rate', 0.5),
            'over_1_5_prob': self._calculate_over_under_prob(total_goals, 1.5),
            'over_2_5_prob': self._calculate_over_under_prob(total_goals, 2.5),
            'over_3_5_prob': self._calculate_over_under_prob(total_goals, 3.5),
        }
    
    def _create_availability_features(self, data: Dict) -> Dict:
        """Player availability features"""
        home_avail = data.get('home_availability', {})
        away_avail = data.get('away_availability', {})
        
        return {
            'home_availability_score': home_avail.get('availability_score', 1.0),
            'away_availability_score': away_avail.get('availability_score', 1.0),
            'availability_diff': (
                home_avail.get('availability_score', 1.0) - 
                away_avail.get('availability_score', 1.0)
            ),
            'home_key_players_missing': home_avail.get('key_players_missing', 0),
            'away_key_players_missing': away_avail.get('key_players_missing', 0),
        }
    
    def _create_home_advantage_features(self, data: Dict) -> Dict:
        """Home advantage features"""
        home = data['home_stats']
        away = data['away_stats']
        
        return {
            'home_advantage_factor': home.get('home_points_per_game', 0) - away.get('away_points_per_game', 0),
            'home_home_win_rate': home.get('home_win_rate', 0),
            'away_away_win_rate': away.get('away_win_rate', 0),
            'home_venue_strength': home.get('home_win_rate', 0) - home.get('away_win_rate', 0),
            'away_travel_weakness': away.get('home_win_rate', 0) - away.get('away_win_rate', 0),
        }
    
    def _create_trend_features(self, data: Dict) -> Dict:
        """Trend features"""
        home_form = data.get('home_form', {})
        away_form = data.get('away_form', {})
        home_stats = data['home_stats']
        away_stats = data['away_stats']
        
        return {
            'home_scoring_trend': (
                home_form.get('avg_goals_scored', 0) - 
                home_stats.get('avg_goals_scored', 0)
            ),
            'away_scoring_trend': (
                away_form.get('avg_goals_scored', 0) - 
                away_stats.get('avg_goals_scored', 0)
            ),
            'home_defense_trend': (
                home_stats.get('avg_goals_conceded', 0) -
                home_form.get('avg_goals_conceded', 0)
            ),
            'away_defense_trend': (
                away_stats.get('avg_goals_conceded', 0) -
                away_form.get('avg_goals_conceded', 0)
            ),
            'home_form_vs_season': home_form.get('form_score', 0.5) - (home_stats.get('win_rate', 0) * 0.8),
            'away_form_vs_season': away_form.get('form_score', 0.5) - (away_stats.get('win_rate', 0) * 0.8),
        }
    
    def _calculate_momentum(self, form_data: Dict) -> float:
        """Calculate team momentum"""
        form_score = form_data.get('form_score', 0.5)
        wins = form_data.get('wins', 0)
        num_matches = form_data.get('num_matches', 1)
        
        momentum = form_score * 0.7
        if num_matches > 0:
            win_rate = wins / num_matches
            momentum += win_rate * 0.3
        
        return min(1.0, momentum)
    
    def _calculate_over_under_prob(self, expected_goals: float, threshold: float) -> float:
        """Calculate over/under probability"""
        diff = expected_goals - threshold
        prob = 1 / (1 + np.exp(-diff))
        return prob
    
    def _safe_divide(self, numerator: float, denominator: float, default: float = 1.0) -> float:
        """Safe division"""
        if denominator == 0:
            return default
        return numerator / denominator

