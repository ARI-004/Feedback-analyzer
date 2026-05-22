# Stakeholder Feedback Intelligence

A local Flask web app for processing stakeholder feedback with your trained machine learning models. The backend loads:

- `C:\Users\ARIJIT ATTA\Downloads\vectorizer (1).pkl`
- `C:\Users\ARIJIT ATTA\Downloads\sentiment_model.pkl`
- `C:\Users\ARIJIT ATTA\Downloads\emotion_model.pkl`

## Features

- Paste comments directly or import `.txt` and `.csv` files.
- ML sentiment classification with confidence probabilities.
- Calibrated sentiment correction when the trained model overuses neutral.
- ML emotion classification with confidence probabilities.
- Theme detection for usability, support, communication, speed, reliability, onboarding, and value.
- Concise summary of overall opinion, key themes, and representative concerns.
- Word cloud visualization for all comments or sentiment-specific comments.
- Full review queue showing every evaluated comment, prediction, raw class, and themes.
- JSON report export for sharing or downstream analysis.

## Run

```powershell
python server.py
```

Then open:

```text
http://127.0.0.1:5000
```

## API

Health check:

```text
GET /api/health
```

Analyze comments:

```text
POST /api/analyze
Content-Type: application/json

{
  "rows": [
    { "comment": "The dashboard is easy to use.", "group": "Operations" }
  ]
}
```

## Label Mapping

The provided pickles expose numeric classes only. The current backend maps:

- Sentiment: `0 = negative`, `1 = neutral`, `2 = positive`
- Emotion: `0 = sadness`, `1 = anger`, `2 = joy`

If your training dataset used a different label order, update `SENTIMENT_LABELS` and `EMOTION_LABELS` in `server.py`.

The sentiment model strongly favors class `1`, so the backend applies a calibration layer. It keeps decisive model predictions, but when neutral wins by a weak or biased margin it uses positive and negative keyword evidence plus model probabilities to classify comments as positive, negative, neutral, or mixed.
