# ML Supply Chain Risk Predictor – Unified Web App

## ✨ Overview
This app is a full solution for supply chain risk analytics using advanced ML:
- **Module 1:** Delivery & Delay Risk
- **Module 2:** Demand Forecast/Inventory
- **Module 3:** Customer Churn & Supplier Reliability
- 💬 Integrated AI Chatbot for project/data questions

Evaluators can upload datasets, run ML models, and get interactive predictions and explanations.

---

## 🚀 How to Run

1. **Clone repo and install requirements**
   ```
   pip install -r requirements.txt
   ```
2. **Place all trained models in `/models/` folder** (see models used in each module below)
3. **Start the app**
   ```
   streamlit run streamlit_app.py
   ```

---

## 📊 Sample Datasets Format
- **Module 1:**  `Days for shipping (real), Days for shipment (scheduled), Shipping Mode, Order Item Quantity, ...`
- **Module 2:**  `date, store, item, sales`
- **Module 3:**  `Customer Segment, Type, Category Name, Order Item Quantity, Sales, Order Profit Per Order, ...`

---

## 📚 Model File List

- `models/catboost_delivery_risk.pkl`
- `models/catboost_delay_regression.pkl`
- `models/catboost_customer_churn.pkl`
- `models/catboost_supplier_reliability.pkl`
- `models/prophet_demand_forecast.pkl` (if trained)
- `models/lstm_demand_forecast.h5` (if trained)

---

## ✅ How To Verify Models
- Upload a prepared sample/test CSV for that module
- Run prediction: output = predictions, metrics (accuracy, MAE, R²), charts
- If model is working: you get meaningful, plausible outputs and visuals (not constant or error)
- Matching columns/order is critical!

---

## 🤖 Chatbot
- The simple chatbot in the last tab answers project and module usage questions directly.

---

## 📁 File Structure
```
supply-chain-risk-app/
├── models/
├── data/
├── streamlit_app.py
├── requirements.txt
├── README.md
```

---

## 📞 Support
- Open an issue in this repo for help, or contact the maintainers via GitHub.
