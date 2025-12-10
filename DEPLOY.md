# 🚀 Deploy на Render

## Стъпка 1: Качи в GitHub

```bash
git init
git add .
git commit -m "Football Prediction System"
git remote add origin https://github.com/YOUR_USERNAME/football-prediction-app.git
git push -u origin main
```

## Стъпка 2: Deploy на Render

1. Отиди на https://render.com
2. "New +" → "Web Service"
3. Свържи GitHub репозитория
4. Настройки:
   - **Name**: `football-prediction`
   - **Language**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Environment Variables (опционално):
   - `FOOTBALL_DATA_API_KEY` = твоя API key
6. "Create Web Service"

## Стъпка 3: Готово!

След 2-3 минути приложението ще е достъпно на твоя Render URL.

## Troubleshooting

- **Build fails**: Провери `requirements.txt`
- **App не стартира**: Провери Start Command
- **API не работи**: Добави `FOOTBALL_DATA_API_KEY` в Environment Variables

