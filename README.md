# 🌧️ RainCast AI — Full-Stack Rainfall Prediction System

> Ensemble ML (Logistic Regression + XGBoost + SVC) · FastAPI Backend · Streamlit Frontend

---

## 📁 Project Structure

```
rainfall_prediction/
├── backend/
│   ├── main.py            ← FastAPI application
│   ├── requirements.txt
│   ├── model.pkl          ← (copy your file here)
│   └── scaler.pkl         ← (copy your file here)
│
└── frontend/
    ├── app.py             ← Streamlit application
    └── requirements.txt
```

---

## ⚡ Quick Start

### 1 · Backend Setup

```bash
cd backend

# Copy your model files
cp /path/to/model.pkl  .
cp /path/to/scaler.pkl .

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API will be live at: **http://localhost:8000**  
Swagger docs at:    **http://localhost:8000/docs**

---

### 2 · Frontend Setup (new terminal)

```bash
cd frontend

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
```

Frontend will open at: **http://localhost:8501**

---

## 🔌 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | Model status check |
| GET | `/model-info` | Ensemble metadata |
| GET | `/features` | Feature definitions |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Up to 100 records |

### Example — Single Predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pressure": 1015.0,
    "temperature": 22.5,
    "dewpoint": 14.0,
    "humidity": 75,
    "cloud": 5,
    "sunshine": 3.2,
    "winddirection": 180,
    "windspeed": 25.0
  }'
```

### Example Response

```json
{
  "request_id": "req_1716979200000",
  "timestamp": "2024-05-19T12:00:00",
  "final_prediction": "Rain",
  "rain_probability": 73.5,
  "no_rain_probability": 26.5,
  "confidence_level": "Moderate",
  "individual_models": [
    {"model_name": "Logistic Regression", "prediction": "Rain", "confidence": 71.2},
    {"model_name": "XGBoost",             "prediction": "Rain", "confidence": 78.9},
    {"model_name": "SVC",                 "prediction": "Rain", "confidence": 70.4}
  ],
  "input_features": { "pressure": 1015.0, "..." : "..." },
  "inference_time_ms": 4.2
}
```

---

## 🌟 Features

### Backend (FastAPI)
- ✅ Ensemble soft-voting across 3 models
- ✅ Full Pydantic validation with range checks
- ✅ Batch endpoint (up to 100 records)
- ✅ `/health` and `/model-info` endpoints
- ✅ CORS enabled for frontend connectivity
- ✅ Confidence level classification (High / Moderate / Low)
- ✅ Request IDs + timestamps on every response

### Frontend (Streamlit)
- ✅ Professional dark-gradient sidebar with live sliders
- ✅ Real-time API health indicator
- ✅ Animated gauge + bar chart for probabilities
- ✅ Per-model vote cards
- ✅ Radar chart of input features
- ✅ Batch CSV upload & download results
- ✅ Dataset explorer with histograms & correlation heatmap
- ✅ API documentation tab

---

## 📦 Model Details

| Component | Details |
|-----------|---------|
| Model 1 | Logistic Regression (L2 penalty) |
| Model 2 | XGBoost Classifier |
| Model 3 | SVC (with probability=True) |
| Voting | Soft voting — averaged probabilities |
| Scaler | StandardScaler (8 features) |
| Features | pressure, temperature, dewpoint, humidity, cloud, sunshine, winddirection, windspeed |

---

*Built with ❤️ using FastAPI + Streamlit*
