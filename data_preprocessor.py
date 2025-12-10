"""
Data Preprocessing Module - Cleans and normalizes football data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FootballDataPreprocessor:
    """Preprocesses raw football data for ML models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def preprocess_match_data(self, raw_match_data: Dict) -> Dict:
        """Preprocess raw match data"""
        try:
            logger.info(f"Preprocessing: {raw_match_data['home_team']} vs {raw_match_data['away_team']}")
            
            processed_data = {
                'home_team': raw_match_data['home_team'],
                'away_team': raw_match_data['away_team'],
                'league': raw_match_data.get('league', 'Unknown'),
                'home_stats': self._clean_team_stats(raw_match_data['home_stats']),
                'away_stats': self._clean_team_stats(raw_match_data['away_stats']),
                'h2h_stats': self._clean_h2h_stats(raw_match_data['head_to_head']),
                'home_availability': self._process_player_availability(raw_match_data['home_player_availability']),
                'away_availability': self._process_player_availability(raw_match_data['away_player_availability']),
                'home_form': self._process_recent_form(raw_match_data['home_recent_form']),
                'away_form': self._process_recent_form(raw_match_data['away_recent_form'])
            }
            
            return processed_data
        except Exception as e:
            logger.error(f"Error preprocessing: {e}")
            raise
    
    def _clean_team_stats(self, team_stats: Dict) -> Dict:
        """Clean and calculate derived metrics"""
        matches_played = max(team_stats.get('matches_played', 1), 1)
        
        cleaned = {
            'matches_played': matches_played,
            'wins': team_stats.get('wins', 0),
            'draws': team_stats.get('draws', 0),
            'losses': team_stats.get('losses', 0),
            'goals_scored': team_stats.get('goals_scored', 0),
            'goals_conceded': team_stats.get('goals_conceded', 0),
            'home_wins': team_stats.get('home_wins', 0),
            'home_draws': team_stats.get('home_draws', 0),
            'home_losses': team_stats.get('home_losses', 0),
            'away_wins': team_stats.get('away_wins', 0),
            'away_draws': team_stats.get('away_draws', 0),
            'away_losses': team_stats.get('away_losses', 0),
        }
        
        # Calculate derived metrics
        cleaned['win_rate'] = cleaned['wins'] / matches_played
        cleaned['draw_rate'] = cleaned['draws'] / matches_played
        cleaned['loss_rate'] = cleaned['losses'] / matches_played
        cleaned['avg_goals_scored'] = cleaned['goals_scored'] / matches_played
        cleaned['avg_goals_conceded'] = cleaned['goals_conceded'] / matches_played
        cleaned['goal_difference'] = cleaned['goals_scored'] - cleaned['goals_conceded']
        cleaned['points'] = (cleaned['wins'] * 3) + cleaned['draws']
        cleaned['points_per_game'] = cleaned['points'] / matches_played
        
        home_matches = cleaned['home_wins'] + cleaned['home_draws'] + cleaned['home_losses']
        if home_matches > 0:
            cleaned['home_win_rate'] = cleaned['home_wins'] / home_matches
            cleaned['home_points_per_game'] = (cleaned['home_wins'] * 3 + cleaned['home_draws']) / home_matches
        else:
            cleaned['home_win_rate'] = 0.0
            cleaned['home_points_per_game'] = 0.0
        
        away_matches = cleaned['away_wins'] + cleaned['away_draws'] + cleaned['away_losses']
        if away_matches > 0:
            cleaned['away_win_rate'] = cleaned['away_wins'] / away_matches
            cleaned['away_points_per_game'] = (cleaned['away_wins'] * 3 + cleaned['away_draws']) / away_matches
        else:
            cleaned['away_win_rate'] = 0.0
            cleaned['away_points_per_game'] = 0.0
        
        cleaned['clean_sheet_rate'] = cleaned.get('clean_sheets', 0) / matches_played
        cleaned['scoring_rate'] = 1 - (cleaned.get('failed_to_score', 0) / matches_played)
        
        return cleaned
    
    def _clean_h2h_stats(self, h2h_data: Dict) -> Dict:
        """Clean H2H statistics"""
        total_matches = max(h2h_data.get('total_matches', 1), 1)
        
        cleaned = {
            'total_matches': total_matches,
            'team1_wins': h2h_data.get('team1_wins', 0),
            'draws': h2h_data.get('draws', 0),
            'team2_wins': h2h_data.get('team2_wins', 0),
            'team1_goals': h2h_data.get('team1_goals', 0),
            'team2_goals': h2h_data.get('team2_goals', 0),
            'avg_goals': h2h_data.get('avg_goals_per_match', 2.5),
            'both_teams_scored': h2h_data.get('both_teams_scored', 0),
            'over_2_5_goals': h2h_data.get('over_2_5_goals', 0),
        }
        
        cleaned['team1_win_rate'] = cleaned['team1_wins'] / total_matches
        cleaned['draw_rate'] = cleaned['draws'] / total_matches
        cleaned['team2_win_rate'] = cleaned['team2_wins'] / total_matches
        cleaned['btts_rate'] = cleaned['both_teams_scored'] / total_matches
        cleaned['over_2_5_rate'] = cleaned['over_2_5_goals'] / total_matches
        
        return cleaned
    
    def _process_player_availability(self, player_data: Dict) -> Dict:
        """Process player availability"""
        num_injuries = len(player_data.get('injuries', []))
        num_suspensions = len(player_data.get('suspensions', []))
        
        processed = {
            'num_injuries': num_injuries,
            'num_suspensions': num_suspensions,
            'total_unavailable': num_injuries + num_suspensions,
            'key_players_missing': player_data.get('key_players_missing', 0),
            'squad_strength': player_data.get('squad_strength', 1.0),
        }
        
        impact = 1.0 - (processed['total_unavailable'] * 0.05)
        impact -= (processed['key_players_missing'] * 0.1)
        processed['availability_score'] = max(0.0, min(1.0, impact))
        
        return processed
    
    def _process_recent_form(self, form_data: List[Dict]) -> Dict:
        """Process recent form"""
        if not form_data:
            return {
                'num_matches': 0, 'wins': 0, 'draws': 0, 'losses': 0,
                'goals_scored': 0, 'goals_conceded': 0, 'form_score': 0.5,
                'avg_goals_scored': 0.0, 'avg_goals_conceded': 0.0,
            }
        
        wins = sum(1 for match in form_data if match.get('result') == 'W')
        draws = sum(1 for match in form_data if match.get('result') == 'D')
        losses = sum(1 for match in form_data if match.get('result') == 'L')
        
        goals_scored = sum(match.get('goals_scored', 0) for match in form_data)
        goals_conceded = sum(match.get('goals_conceded', 0) for match in form_data)
        
        num_matches = len(form_data)
        points = (wins * 3) + draws
        max_points = num_matches * 3
        form_score = points / max_points if max_points > 0 else 0.5
        
        return {
            'num_matches': num_matches,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'goals_scored': goals_scored,
            'goals_conceded': goals_conceded,
            'form_score': form_score,
            'avg_goals_scored': goals_scored / num_matches,
            'avg_goals_conceded': goals_conceded / num_matches,
            'win_rate': wins / num_matches,
            'points_per_game': points / num_matches,
        }
    
    def normalize_features(self, features_df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Normalize features"""
        try:
            numerical_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numerical_cols:
                return features_df
            
            normalized_df = features_df.copy()
            
            if fit:
                normalized_df[numerical_cols] = self.scaler.fit_transform(features_df[numerical_cols])
                self.is_fitted = True
            else:
                if not self.is_fitted:
                    return features_df
                normalized_df[numerical_cols] = self.scaler.transform(features_df[numerical_cols])
            
            normalized_df = normalized_df.replace([np.inf, -np.inf], np.nan)
            for col in numerical_cols:
                if normalized_df[col].isna().any():
                    median_val = normalized_df[col].median()
                    normalized_df[col].fillna(median_val if not np.isnan(median_val) else 0, inplace=True)
            
            return normalized_df
        except Exception as e:
            logger.error(f"Error normalizing: {e}")
            return features_df
    
    def create_feature_vector(self, preprocessed_data: Dict) -> np.ndarray:
        """Create feature vector"""
        features = []
        
        home_stats = preprocessed_data['home_stats']
        features.extend([
            home_stats.get('win_rate', 0),
            home_stats.get('draw_rate', 0),
            home_stats.get('avg_goals_scored', 0),
            home_stats.get('avg_goals_conceded', 0),
            home_stats.get('home_win_rate', 0),
            home_stats.get('home_points_per_game', 0),
            home_stats.get('clean_sheet_rate', 0),
            home_stats.get('scoring_rate', 0),
        ])
        
        away_stats = preprocessed_data['away_stats']
        features.extend([
            away_stats.get('win_rate', 0),
            away_stats.get('draw_rate', 0),
            away_stats.get('avg_goals_scored', 0),
            away_stats.get('avg_goals_conceded', 0),
            away_stats.get('away_win_rate', 0),
            away_stats.get('away_points_per_game', 0),
            away_stats.get('clean_sheet_rate', 0),
            away_stats.get('scoring_rate', 0),
        ])
        
        h2h = preprocessed_data.get('h2h_stats', {})
        features.extend([
            h2h.get('team1_win_rate', 0.33),
            h2h.get('draw_rate', 0.33),
            h2h.get('avg_goals', 2.5),
            h2h.get('btts_rate', 0.5),
            h2h.get('over_2_5_rate', 0.5),
        ])
        
        home_form = preprocessed_data.get('home_form', {})
        away_form = preprocessed_data.get('away_form', {})
        features.extend([
            home_form.get('form_score', 0.5),
            home_form.get('avg_goals_scored', 0),
            home_form.get('avg_goals_conceded', 0),
            away_form.get('form_score', 0.5),
            away_form.get('avg_goals_scored', 0),
            away_form.get('avg_goals_conceded', 0),
        ])
        
        home_avail = preprocessed_data.get('home_availability', {})
        away_avail = preprocessed_data.get('away_availability', {})
        features.extend([
            home_avail.get('availability_score', 1.0),
            away_avail.get('availability_score', 1.0),
        ])
        
        return np.array(features)

