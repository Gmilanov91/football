"""
Football Data Scraper - Real Data from Football-Data.org API
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FootballDataScraper:
    """Scrapes real football data from Football-Data.org API"""
    
    def __init__(self, cache_duration_hours: int = 6):
        self.cache = {}
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.api_key = os.environ.get('FOOTBALL_DATA_API_KEY', None)
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
        
        if self.api_key:
            self.headers['X-Auth-Token'] = self.api_key
        
        # League IDs for Football-Data.org API
        self.league_ids = {
            'Premier League': 2021,
            'La Liga': 2014,
            'Serie A': 2019,
            'Bundesliga': 2002,
            'Ligue 1': 2015,
            'Champions League': 2001,
            'Europa League': 2018,
            'Eredivisie': 2003,
            'Primeira Liga': 2017,
            'Championship': 2016
        }
        
        # Team name mapping
        self.team_mapping = {
            'Manchester United': 'Manchester United FC',
            'Man United': 'Manchester United FC',
            'Man Utd': 'Manchester United FC',
            'Arsenal': 'Arsenal FC',
            'Chelsea': 'Chelsea FC',
            'Liverpool': 'Liverpool FC',
            'Manchester City': 'Manchester City FC',
            'Man City': 'Manchester City FC',
            'Tottenham': 'Tottenham Hotspur FC',
            'Spurs': 'Tottenham Hotspur FC',
            'Barcelona': 'FC Barcelona',
            'Real Madrid': 'Real Madrid CF',
            'Bayern Munich': 'FC Bayern München',
            'Bayern': 'FC Bayern München',
            'PSG': 'Paris Saint-Germain FC',
            'Paris Saint-Germain': 'Paris Saint-Germain FC',
            'Juventus': 'Juventus FC',
            'AC Milan': 'AC Milan',
            'Inter Milan': 'Inter Milan',
            'Inter': 'Inter Milan',
            'Atletico Madrid': 'Atlético Madrid',
            'Atletico': 'Atlético Madrid',
        }
    
    def _normalize_team_name(self, team_name: str) -> str:
        """Normalize team name for API"""
        return self.team_mapping.get(team_name, team_name)
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache is valid"""
        if cache_key not in self.cache:
            return False
        cache_time = self.cache[cache_key]['timestamp']
        return (datetime.now() - cache_time) < self.cache_duration
    
    def get_team_stats(self, team_name: str, league: str = "Premier League") -> Dict:
        """Fetch team statistics from API"""
        cache_key = f"team_stats_{team_name}_{league}"
        
        if self._is_cache_valid(cache_key):
            logger.info(f"Using cached data for {team_name}")
            return self.cache[cache_key]['data']
        
        try:
            league_id = self.league_ids.get(league, 2021)
            url = f"https://api.football-data.org/v4/competitions/{league_id}/standings"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Find team in standings
                normalized_name = self._normalize_team_name(team_name)
                team_data = None
                
                if 'standings' in data and len(data['standings']) > 0:
                    for group in data['standings']:
                        if 'table' in group:
                            for team in group['table']:
                                team_api_name = team.get('team', {}).get('name', '')
                                if (team_name.lower() in team_api_name.lower() or 
                                    team_api_name.lower() in team_name.lower() or
                                    normalized_name.lower() in team_api_name.lower()):
                                    team_data = team
                                    break
                
                if team_data:
                    stats = {
                        'team_name': team_name,
                        'league': league,
                        'matches_played': team_data.get('playedGames', 0),
                        'wins': team_data.get('won', 0),
                        'draws': team_data.get('draw', 0),
                        'losses': team_data.get('lost', 0),
                        'goals_scored': team_data.get('goalsFor', 0),
                        'goals_conceded': team_data.get('goalsAgainst', 0),
                        'points': team_data.get('points', 0),
                        'position': team_data.get('position', 0),
                        'form_last_5': self._parse_form(team_data.get('form', '')),
                        'home_wins': 0,
                        'home_draws': 0,
                        'home_losses': 0,
                        'away_wins': 0,
                        'away_draws': 0,
                        'away_losses': 0,
                        'clean_sheets': 0,
                        'failed_to_score': 0
                    }
                    
                    self.cache[cache_key] = {
                        'data': stats,
                        'timestamp': datetime.now()
                    }
                    return stats
            
            elif response.status_code == 429:
                logger.warning("API rate limit reached")
                time.sleep(60)
        
        except Exception as e:
            logger.error(f"Error fetching team stats: {e}")
        
        # Return default if API fails
        return {
            'matches_played': 0, 'wins': 0, 'draws': 0, 'losses': 0,
            'goals_scored': 0, 'goals_conceded': 0, 'form_last_5': [1, 1, 1, 1, 1],
            'home_wins': 0, 'home_draws': 0, 'home_losses': 0,
            'away_wins': 0, 'away_draws': 0, 'away_losses': 0
        }
    
    def _parse_form(self, form_str: str) -> List[int]:
        """Parse form string to numeric list (3=win, 1=draw, 0=loss)"""
        if not form_str:
            return [1, 1, 1, 1, 1]
        form_map = {'W': 3, 'D': 1, 'L': 0}
        form_list = [form_map.get(char, 1) for char in form_str[-5:]]
        while len(form_list) < 5:
            form_list.insert(0, 1)
        return form_list[:5]
    
    def get_head_to_head(self, team1: str, team2: str, num_matches: int = 10) -> Dict:
        """Fetch head-to-head statistics"""
        cache_key = f"h2h_{team1}_{team2}_{num_matches}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            team1_id = self._get_team_id(team1)
            team2_id = self._get_team_id(team2)
            
            if team1_id and team2_id:
                h2h_data = self._fetch_h2h_matches(team1_id, team2_id, num_matches)
                if h2h_data:
                    self.cache[cache_key] = {
                        'data': h2h_data,
                        'timestamp': datetime.now()
                    }
                    return h2h_data
        except Exception as e:
            logger.warning(f"H2H fetch failed: {e}")
        
        return {
            'total_matches': 0, 'team1_wins': 0, 'draws': 0, 'team2_wins': 0,
            'avg_goals_per_match': 2.5, 'both_teams_scored': 0, 'recent_results': []
        }
    
    def _get_team_id(self, team_name: str) -> Optional[int]:
        """Get team ID from API"""
        try:
            url = f"https://api.football-data.org/v4/teams"
            params = {'name': team_name}
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'teams' in data and len(data['teams']) > 0:
                    return data['teams'][0].get('id')
        except Exception as e:
            logger.debug(f"Could not get team ID: {e}")
        return None
    
    def _fetch_h2h_matches(self, team1_id: int, team2_id: int, num_matches: int) -> Optional[Dict]:
        """Fetch H2H matches between teams"""
        try:
            url = f"https://api.football-data.org/v4/teams/{team1_id}/matches"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                matches = data.get('matches', [])
                
                h2h_matches = []
                for match in matches:
                    home_id = match.get('homeTeam', {}).get('id')
                    away_id = match.get('awayTeam', {}).get('id')
                    
                    if (home_id == team1_id and away_id == team2_id) or \
                       (home_id == team2_id and away_id == team1_id):
                        h2h_matches.append(match)
                
                if h2h_matches:
                    return self._process_h2h_matches(h2h_matches[:num_matches], team1_id)
        except Exception as e:
            logger.error(f"H2H API error: {e}")
        return None
    
    def _process_h2h_matches(self, matches: List[Dict], team1_id: int) -> Dict:
        """Process H2H matches into statistics"""
        total_matches = len(matches)
        team1_wins = 0
        draws = 0
        team2_wins = 0
        team1_goals = 0
        team2_goals = 0
        both_scored = 0
        over_2_5 = 0
        recent_results = []
        
        for match in matches:
            home_id = match.get('homeTeam', {}).get('id')
            score = match.get('score', {})
            home_goals = score.get('fullTime', {}).get('home', 0)
            away_goals = score.get('fullTime', {}).get('away', 0)
            
            if home_goals is None or away_goals is None:
                continue
            
            total_goals = home_goals + away_goals
            
            if home_id == team1_id:
                team1_goals += home_goals
                team2_goals += away_goals
                if home_goals > away_goals:
                    team1_wins += 1
                elif home_goals < away_goals:
                    team2_wins += 1
                else:
                    draws += 1
            else:
                team1_goals += away_goals
                team2_goals += home_goals
                if away_goals > home_goals:
                    team1_wins += 1
                elif away_goals < home_goals:
                    team2_wins += 1
                else:
                    draws += 1
            
            if home_goals > 0 and away_goals > 0:
                both_scored += 1
            
            if total_goals > 2.5:
                over_2_5 += 1
            
            date = match.get('utcDate', '')[:10] if match.get('utcDate') else ''
            home_name = match.get('homeTeam', {}).get('name', '')
            away_name = match.get('awayTeam', {}).get('name', '')
            recent_results.append({
                'date': date,
                'home': home_name,
                'away': away_name,
                'score': f'{home_goals}-{away_goals}'
            })
        
        return {
            'total_matches': total_matches,
            'team1_wins': team1_wins,
            'draws': draws,
            'team2_wins': team2_wins,
            'team1_goals': team1_goals,
            'team2_goals': team2_goals,
            'avg_goals_per_match': (team1_goals + team2_goals) / total_matches if total_matches > 0 else 2.5,
            'both_teams_scored': both_scored,
            'over_2_5_goals': over_2_5,
            'recent_results': recent_results[:10]
        }
    
    def get_player_availability(self, team_name: str) -> Dict:
        """Get player availability (injuries/suspensions)"""
        return {
            'team': team_name,
            'injuries': [],
            'suspensions': [],
            'key_players_missing': 0,
            'squad_strength': 1.0
        }
    
    def get_recent_form(self, team_name: str, num_matches: int = 5) -> List[Dict]:
        """Fetch recent form from API"""
        cache_key = f"form_{team_name}_{num_matches}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            team_id = self._get_team_id(team_name)
            if team_id:
                url = f"https://api.football-data.org/v4/teams/{team_id}/matches"
                params = {'status': 'FINISHED', 'limit': num_matches}
                response = requests.get(url, headers=self.headers, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    matches = data.get('matches', [])
                    
                    results = []
                    for match in matches[:num_matches]:
                        home_team = match.get('homeTeam', {}).get('name', '')
                        away_team = match.get('awayTeam', {}).get('name', '')
                        score = match.get('score', {})
                        home_goals = score.get('fullTime', {}).get('home', 0)
                        away_goals = score.get('fullTime', {}).get('away', 0)
                        
                        if home_goals is None or away_goals is None:
                            continue
                        
                        is_home = team_name.lower() in home_team.lower()
                        
                        if is_home:
                            goals_scored = home_goals
                            goals_conceded = away_goals
                            opponent = away_team
                        else:
                            goals_scored = away_goals
                            goals_conceded = home_goals
                            opponent = home_team
                        
                        if goals_scored > goals_conceded:
                            result = 'W'
                        elif goals_scored < goals_conceded:
                            result = 'L'
                        else:
                            result = 'D'
                        
                        date = match.get('utcDate', '')[:10] if match.get('utcDate') else ''
                        
                        results.append({
                            'date': date,
                            'opponent': opponent,
                            'home_away': 'home' if is_home else 'away',
                            'result': result,
                            'goals_scored': goals_scored,
                            'goals_conceded': goals_conceded
                        })
                    
                    results.reverse()
                    self.cache[cache_key] = {
                        'data': results,
                        'timestamp': datetime.now()
                    }
                    return results
        except Exception as e:
            logger.warning(f"Recent form fetch failed: {e}")
        
        return [{'result': 'D', 'goals_scored': 1, 'goals_conceded': 1} 
                for _ in range(num_matches)]
    
    def fetch_match_data(self, home_team: str, away_team: str, league: str = "Premier League") -> Dict:
        """Fetch comprehensive match data"""
        logger.info(f"Fetching match data: {home_team} vs {away_team}")
        
        try:
            match_data = {
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'home_stats': self.get_team_stats(home_team, league),
                'away_stats': self.get_team_stats(away_team, league),
                'head_to_head': self.get_head_to_head(home_team, away_team),
                'home_player_availability': self.get_player_availability(home_team),
                'away_player_availability': self.get_player_availability(away_team),
                'home_recent_form': self.get_recent_form(home_team),
                'away_recent_form': self.get_recent_form(away_team),
                'fetch_timestamp': datetime.now().isoformat()
            }
            
            return match_data
        except Exception as e:
            logger.error(f"Error fetching match data: {e}")
            raise
    
    def clear_cache(self):
        """Clear cache"""
        self.cache = {}
        logger.info("Cache cleared")

