"""
Flask Web Application for Football Prediction System
Optimized for Render deployment
"""

from flask import Flask, render_template_string, jsonify, request
from predictor import FootballPredictor
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
predictor = None

def get_predictor():
    """Lazy load predictor"""
    global predictor
    if predictor is None:
        try:
            predictor = FootballPredictor(model_type='mlp')
            logger.info("Predictor initialized")
        except Exception as e:
            logger.error(f"Error initializing predictor: {e}")
    return predictor

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚽ Football Match Prediction</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }
        h1 { color: #333; text-align: center; margin-bottom: 10px; font-size: 2.5em; }
        .subtitle { text-align: center; color: #666; margin-bottom: 30px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; color: #333; font-weight: 600; }
        input[type="text"], select {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
        }
        input:focus, select:focus { outline: none; border-color: #667eea; }
        .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }
        button:disabled { opacity: 0.6; cursor: not-allowed; }
        .result {
            margin-top: 30px;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 12px;
            display: none;
        }
        .result.show { display: block; animation: fadeIn 0.5s; }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .prediction-box {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
        }
        .prediction-box h3 { color: #667eea; margin-bottom: 15px; }
        .probabilities {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin: 15px 0;
        }
        .prob-item {
            text-align: center;
            padding: 15px;
            background: #f0f0f0;
            border-radius: 8px;
        }
        .prob-item .label { font-size: 0.9em; color: #666; margin-bottom: 5px; }
        .prob-item .value { font-size: 1.5em; font-weight: bold; color: #333; }
        .goals {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin: 15px 0;
        }
        .goal-item {
            text-align: center;
            padding: 15px;
            background: #e8f4f8;
            border-radius: 8px;
        }
        .goal-item .label { font-size: 0.9em; color: #666; margin-bottom: 5px; }
        .goal-item .value { font-size: 1.8em; font-weight: bold; color: #2196F3; }
        .betting-markets {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin: 15px 0;
        }
        .market-item {
            padding: 15px;
            background: #fff3cd;
            border-radius: 8px;
            text-align: center;
        }
        .market-item .label { font-size: 0.9em; color: #666; margin-bottom: 5px; }
        .market-item .value { font-size: 1.3em; font-weight: bold; color: #856404; }
        .loading {
            text-align: center;
            padding: 20px;
            color: #667eea;
            display: none;
        }
        .loading.show { display: block; }
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            display: none;
        }
        .error.show { display: block; }
        .key-factors {
            margin-top: 20px;
            padding: 15px;
            background: #d1ecf1;
            border-radius: 8px;
        }
        .key-factors h4 { color: #0c5460; margin-bottom: 10px; }
        .key-factors ul { list-style: none; padding: 0; }
        .key-factors li { padding: 5px 0; color: #0c5460; }
        .key-factors li:before { content: "✓ "; color: #28a745; font-weight: bold; }
        @media (max-width: 768px) {
            .form-row { grid-template-columns: 1fr; }
            .probabilities, .goals, .betting-markets { grid-template-columns: 1fr; }
            .container { padding: 20px; }
            h1 { font-size: 2em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚽ Football Match Prediction</h1>
        <p class="subtitle">AI-Powered Match Outcome Predictions</p>
        
        <form id="predictForm">
            <div class="form-row">
                <div class="form-group">
                    <label for="homeTeam">Home Team</label>
                    <input type="text" id="homeTeam" placeholder="e.g., Arsenal" required>
                </div>
                <div class="form-group">
                    <label for="awayTeam">Away Team</label>
                    <input type="text" id="awayTeam" placeholder="e.g., Chelsea" required>
                </div>
            </div>
            
            <div class="form-group">
                <label for="league">League</label>
                <select id="league">
                    <option>Premier League</option>
                    <option>La Liga</option>
                    <option>Serie A</option>
                    <option>Bundesliga</option>
                    <option>Ligue 1</option>
                    <option>Champions League</option>
                    <option>Europa League</option>
                </select>
            </div>
            
            <button type="submit" id="submitBtn">Predict Match Outcome</button>
        </form>
        
        <div class="loading" id="loading">
            <p>🔮 Analyzing match data and making prediction...</p>
        </div>
        
        <div class="error" id="error"></div>
        
        <div class="result" id="result">
            <h2>Prediction Results</h2>
            <div id="resultContent"></div>
        </div>
    </div>
    
    <script>
        document.getElementById('predictForm').onsubmit = async (e) => {
            e.preventDefault();
            const home = document.getElementById('homeTeam').value.trim();
            const away = document.getElementById('awayTeam').value.trim();
            const league = document.getElementById('league').value;
            
            document.getElementById('loading').classList.add('show');
            document.getElementById('result').classList.remove('show');
            document.getElementById('error').classList.remove('show');
            document.getElementById('submitBtn').disabled = true;
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({home, away, league})
                });
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                displayResults(data.prediction, home, away);
                
            } catch (error) {
                document.getElementById('error').textContent = 
                    'Error: ' + error.message + '. Please try again.';
                document.getElementById('error').classList.add('show');
            } finally {
                document.getElementById('loading').classList.remove('show');
                document.getElementById('submitBtn').disabled = false;
            }
        };
        
        function displayResults(prediction, homeTeam, awayTeam) {
            const result = prediction.match_result;
            const goals = prediction.expected_goals;
            const betting = prediction.betting_markets;
            const analysis = prediction.analysis || {};
            const factors = analysis.key_factors || [];
            
            const html = `
                <div class="prediction-box">
                    <h3>🎯 Predicted Outcome: ${result.predicted_outcome}</h3>
                    <p style="color: #666; margin-bottom: 15px;">
                        Confidence: ${(result.confidence * 100).toFixed(1)}%
                    </p>
                    
                    <div class="probabilities">
                        <div class="prob-item">
                            <div class="label">${homeTeam} Win</div>
                            <div class="value">${(result.home_win_probability * 100).toFixed(1)}%</div>
                        </div>
                        <div class="prob-item">
                            <div class="label">Draw</div>
                            <div class="value">${(result.draw_probability * 100).toFixed(1)}%</div>
                        </div>
                        <div class="prob-item">
                            <div class="label">${awayTeam} Win</div>
                            <div class="value">${(result.away_win_probability * 100).toFixed(1)}%</div>
                        </div>
                    </div>
                </div>
                
                <div class="prediction-box">
                    <h3>⚽ Expected Goals</h3>
                    <div class="goals">
                        <div class="goal-item">
                            <div class="label">${homeTeam}</div>
                            <div class="value">${goals.home.toFixed(2)}</div>
                        </div>
                        <div class="goal-item">
                            <div class="label">Total</div>
                            <div class="value">${goals.total.toFixed(2)}</div>
                        </div>
                        <div class="goal-item">
                            <div class="label">${awayTeam}</div>
                            <div class="value">${goals.away.toFixed(2)}</div>
                        </div>
                    </div>
                </div>
                
                <div class="prediction-box">
                    <h3>🎲 Betting Markets</h3>
                    <div class="betting-markets">
                        <div class="market-item">
                            <div class="label">Over 2.5 Goals</div>
                            <div class="value">${(betting.over_2_5_probability * 100).toFixed(1)}%</div>
                        </div>
                        <div class="market-item">
                            <div class="label">Under 2.5 Goals</div>
                            <div class="value">${(betting.under_2_5_probability * 100).toFixed(1)}%</div>
                        </div>
                        <div class="market-item">
                            <div class="label">Both Teams Score</div>
                            <div class="value">${(betting.btts_probability * 100).toFixed(1)}%</div>
                        </div>
                        <div class="market-item">
                            <div class="label">Clean Sheet</div>
                            <div class="value">${(betting.btts_no_probability * 100).toFixed(1)}%</div>
                        </div>
                    </div>
                </div>
                
                ${factors.length > 0 ? `
                <div class="key-factors">
                    <h4>🔍 Key Factors</h4>
                    <ul>
                        ${factors.map(factor => `<li>${factor}</li>`).join('')}
                    </ul>
                </div>
                ` : ''}
            `;
            
            document.getElementById('resultContent').innerHTML = html;
            document.getElementById('result').classList.add('show');
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    """Home page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction API endpoint"""
    try:
        data = request.json
        
        if not data or 'home' not in data or 'away' not in data:
            return jsonify({'error': 'Missing required fields: home and away'}), 400
        
        home_team = data['home']
        away_team = data['away']
        league = data.get('league', 'Premier League')
        
        logger.info(f"Prediction request: {home_team} vs {away_team} ({league})")
        
        pred = get_predictor()
        if pred is None:
            return jsonify({'error': 'Predictor not initialized'}), 500
        
        prediction = pred.predict_match(
            home_team=home_team,
            away_team=away_team,
            league=league,
            return_details=True
        )
        
        return jsonify({'prediction': prediction})
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Football Prediction System',
        'predictor_initialized': predictor is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

