import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import google.generativeai as genai
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY") or "YOUR_GEMINI_API_KEY"

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro")

st.set_page_config(page_title="Supply Chain AI Dashboard", layout="wide")

if 'uploaded_data' not in st.session_state:
    st.session_state['uploaded_data'] = {}

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

st.sidebar.title("Navigation")
selected = st.sidebar.radio("Choose Module:", ["Home", "Supply Chain Gemini Chatbot"])

if selected == "Home":
    st.title("Supply Chain Risk Predictor")
    st.markdown("Upload your data and use the Gemini chatbot for predictions/explanations.")

if selected == "Supply Chain Gemini Chatbot":
    st.title("Supply Chain Chatbot (Gemini + Real Models)")
    tabs = st.tabs(["Delivery", "Demand", "Churn/Supplier"])
    labels = ["delivery", "demand", "churn"]

    for i, tab in enumerate(tabs):
        with tab:
            label = labels[i]
            upl = st.file_uploader(f"Upload {label.capitalize()} Data", key=label)
            if upl:
                df_i = pd.read_csv(upl) if upl.name.endswith('.csv') else pd.read_excel(upl)
                st.session_state['uploaded_data'][label] = df_i
                st.write(df_i.head())

    st.subheader("Ask your supply chain AI chatbot (Gemini 2.5)")
    user_input = st.text_input("Your question:")

    def call_project_models(user_input):
        # Intent routing, as before
        if "late delivery" in user_input.lower():
            df = st.session_state["uploaded_data"].get("delivery")
            if df is not None:
                model_risk = joblib.load("models/catboost_delivery_risk.pkl")
                cols = ['Days for shipping (real)', 'Days for shipment (scheduled)', 'Shipping Mode', 'Order Item Quantity']
                pred = model_risk.predict(df[cols])
                return f"Late delivery risk predictions: {list(pred)}"
            else:
                return "Please upload delivery data."
        elif "delay days" in user_input.lower() or "how many days" in user_input.lower():
            df = st.session_state["uploaded_data"].get("delivery")
            if df is not None:
                model_delay = joblib.load("models/catboost_delay_regression.pkl")
                cols = ['Days for shipping (real)', 'Days for shipment (scheduled)', 'Shipping Mode', 'Order Item Quantity']
                pred = model_delay.predict(df[cols])
                return f"Predicted delay (in days): {list(pred)}"
            else:
                return "Please upload delivery data."
        elif "demand" in user_input.lower() or "forecast" in user_input.lower():
            df = st.session_state["uploaded_data"].get("demand")
            if df is not None:
                from prophet import Prophet
                df['date'] = pd.to_datetime(df['date'])
                day_sales = df.groupby("date")["sales"].sum().reset_index()
                prophet_df = day_sales.rename(columns={"date": "ds", "sales": "y"})
                model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                model_prophet.fit(prophet_df)
                future = model_prophet.make_future_dataframe(periods=30)
                forecast = model_prophet.predict(future)
                last_days = forecast[["ds", "yhat"]].tail(30)
                return f"Demand forecast:\n{last_days.to_dict(orient='records')}"
            else:
                return "Please upload demand data."
        elif "churn" in user_input.lower():
            df = st.session_state["uploaded_data"].get("churn")
            if df is not None:
                model_churn = joblib.load("models/catboost_customer_churn.pkl")
                cols = [ 'Customer Segment', 'Type', 'Category Name', 'Order Item Quantity', 'Sales', 'Order Profit Per Order']
                pred = model_churn.predict(df[cols])
                return f"Churn predictions: {list(pred)}"
            else:
                return "Please upload customer churn data."
        elif "reliability" in user_input.lower() or "supplier" in user_input.lower():
            df = st.session_state["uploaded_data"].get("churn")
            if df is not None:
                model_supplier = joblib.load("models/catboost_supplier_reliability.pkl")
                cols = ['Order Item Quantity', 'Order Profit Per Order', 'Sales']
                pred = model_supplier.predict(df[cols])
                return f"Supplier reliability scores: {list(pred)}"
            else:
                return "Please upload supplier data."
        else:
            return None

    if user_input and GEMINI_API_KEY and model is not None:
        project_response = call_project_models(user_input)
        if project_response:
            prompt = (f"User asked: {user_input}\n"
                      f"Supply Chain AI predicts: {project_response}\n"
                      f"Summarize this result clearly and concisely for the user.")
        else:
            prompt = (f"User asked: {user_input}\n"
                      f"No relevant data uploaded or query not matched. Request user to upload or clarify.")

        gemini_response = model.generate_content(prompt).text.strip()
        st.session_state['chat_history'].append({"role":"user", "content":user_input})
        st.session_state['chat_history'].append({"role":"assistant", "content":gemini_response})
        st.write("AI:", gemini_response)

    if st.session_state['chat_history']:
        st.subheader("Chat History")
        for entry in st.session_state['chat_history']:
            role_icon = "👤" if entry["role"]=="user" else "🤖"
            st.markdown(f"{role_icon} {entry['content']}")

