# ⚽ Football Match Prediction System

AI-powered football match prediction system using Neural Networks and real-time data from Football-Data.org API.

## Features

- **Real-time Data**: Fetches live statistics from Football-Data.org API
- **Neural Network Predictions**: MLP model for accurate match predictions
- **Multi-Output Predictions**: 
  - Match result (Home Win, Draw, Away Win) with probabilities
  - Expected goals for each team
  - Over/Under goals probability
  - Both Teams to Score (BTTS)
- **Web Interface**: Beautiful, responsive Flask web application
- **Cloud Ready**: Optimized for Render deployment

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

Visit `http://localhost:5000` to use the web interface.

### Deploy on Render

1. Push code to GitHub
2. Connect repository in Render
3. Set environment variables (optional):
   - `FOOTBALL_DATA_API_KEY` - Your API key from football-data.org
4. Deploy!

## API Usage

### Web Interface
Visit the homepage and enter:
- Home Team
- Away Team
- League

### API Endpoint

```bash
POST /predict
Content-Type: application/json

{
  "home": "Arsenal",
  "away": "Chelsea",
  "league": "Premier League"
}
```

### Health Check

```bash
GET /health
```

## System Architecture

- `data_scraper.py` - Real data fetching from Football-Data.org API
- `data_preprocessor.py` - Data cleaning and normalization
- `feature_engineer.py` - Advanced feature engineering (60+ features)
- `neural_model.py` - Neural Network model (MLP)
- `predictor.py` - Main prediction engine
- `app.py` - Flask web application

## Requirements

- Python 3.11+
- See `requirements.txt` for dependencies

## API Key (Optional)

Get a free API key from https://www.football-data.org/client/register

Set as environment variable:
```bash
export FOOTBALL_DATA_API_KEY=your_key_here
```

Or in Render: Add as Environment Variable in dashboard.

## Supported Leagues

- Premier League
- La Liga
- Serie A
- Bundesliga
- Ligue 1
- Champions League
- Europa League
- And more...

## License

MIT License

