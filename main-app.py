import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from prophet import Prophet
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import google.generativeai as genai

# Configure Gemini API
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro")
else:
    model = None
    st.warning("⚠️ Gemini API key not found. Chatbot will be limited.")

st.set_page_config(page_title="Supply Chain AI Dashboard", layout="wide")

# Session state for data persistence
if 'uploaded_data' not in st.session_state:
    st.session_state['uploaded_data'] = {}
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

st.sidebar.title("Navigation")
selected = st.sidebar.radio("Choose Module:", ["Home", "Supply Chain Smart Chatbot"])

# ---- HOME PAGE ----
if selected == "Home":
    st.title("🏭 Supply Chain Risk Predictor")
    st.markdown("""
    Welcome to your integrated ML-powered supply chain dashboard!
    
    **Features:**
    - 📦 **Module 1:** Delivery & Delay Risk Prediction
    - 📈 **Module 2:** Demand/Inventory Forecasting  
    - 👥 **Module 3:** Customer Churn & Supplier Reliability
    - 🤖 **Gemini Chatbot:** Ask questions and get AI-powered insights from real models
    
    Navigate to the **"Supply Chain Smart Chatbot"** tab to get started!
    """)

# ---- CHATBOT + MODULE TABS ----
if selected == "Supply Chain Smart Chatbot":
    st.title("🤖 Supply Chain Smart Chatbot (Gemini 2.5 + Real Models)")
    
    # Tab interface for data upload
    tabs = st.tabs(["Delivery", "Demand", "Churn/Supplier"])
    labels = ["delivery", "demand", "churn"]
    
    for i, tab in enumerate(tabs):
        with tab:
            label = labels[i]
            upl = st.file_uploader(f"Upload {label.capitalize()} Data", key=label, type=['csv', 'xlsx'])
            if upl:
                df_i = pd.read_csv(upl) if upl.name.endswith('.csv') else pd.read_excel(upl)
                st.session_state['uploaded_data'][label] = df_i
                st.write(f"✅ **{label.capitalize()} data loaded:**")
                st.write(df_i.head())
    
    # ---- CHATBOT INTERFACE ----
    st.subheader("Ask your supply chain AI chatbot (Gemini 2.5 + Real ML Models)")
    st.info("Upload data in the tabs above, then ask questions about predictions!")
    
    user_input = st.text_input("Your question:", placeholder="e.g., 'What is the late delivery risk?'")
    
    def call_project_models(user_input):
        """Route user query to appropriate ML model and return prediction."""
        if not user_input:
            return None
        
        user_lower = user_input.lower()
        
        # Module 1: Late Delivery Risk
        if "late delivery" in user_lower or "delivery risk" in user_lower:
            df = st.session_state["uploaded_data"].get("delivery")
            if df is not None:
                try:
                    model_risk = joblib.load("catboost_delivery_risk.pkl")
                    cols = ['Days for shipping (real)', 'Days for shipment (scheduled)', 'Shipping Mode', 'Order Item Quantity']
                    if all(col in df.columns for col in cols):
                        pred = model_risk.predict(df[cols])
                        return f"Late delivery risk predictions: {list(pred[:5])} (showing first 5)"
                    else:
                        return "Missing required columns for delivery risk model."
                except Exception as e:
                    return f"Error loading delivery model: {str(e)}"
            else:
                return "Please upload delivery data in the 'Delivery' tab first."
        
        # Module 1: Delay in Days
        elif "delay days" in user_lower or "how many days" in user_lower:
            df = st.session_state["uploaded_data"].get("delivery")
            if df is not None:
                try:
                    model_delay = joblib.load("catboost_delay_regression.pkl")
                    cols = ['Days for shipping (real)', 'Days for shipment (scheduled)', 'Shipping Mode', 'Order Item Quantity']
                    if all(col in df.columns for col in cols):
                        pred = model_delay.predict(df[cols])
                        return f"Predicted delay (in days): {list(pred[:5])} (showing first 5)"
                    else:
                        return "Missing required columns for delay model."
                except Exception as e:
                    return f"Error loading delay model: {str(e)}"
            else:
                return "Please upload delivery data in the 'Delivery' tab first."
        
        # Module 2: Demand Forecast - Prophet
        elif "demand" in user_lower or "forecast" in user_lower:
            df = st.session_state["uploaded_data"].get("demand")
            if df is not None:
                try:
                    if 'date' in df.columns and 'sales' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        day_sales = df.groupby('date')['sales'].sum().reset_index()
                        prophet_df = day_sales.rename(columns={"date": "ds", "sales": "y"})
                        model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, interval_width=0.95)
                        model_prophet.fit(prophet_df)
                        future = model_prophet.make_future_dataframe(periods=7)
                        forecast = model_prophet.predict(future)
                        last_days = forecast[["ds", "yhat"]].tail(7)
                        return f"7-day demand forecast:\n{last_days.to_string()}"
                    else:
                        return "Demand data must have 'date' and 'sales' columns."
                except Exception as e:
                    return f"Error in demand forecasting: {str(e)}"
            else:
                return "Please upload demand data in the 'Demand' tab first."
        
        # Module 3: Customer Churn
        elif "churn" in user_lower:
            df = st.session_state["uploaded_data"].get("churn")
            if df is not None:
                try:
                    model_churn = joblib.load("catboost_customer_churn.pkl")
                    cols = ['Customer Segment', 'Type', 'Category Name', 'Order Item Quantity', 'Sales', 'Order Profit Per Order']
                    if all(col in df.columns for col in cols):
                        pred = model_churn.predict(df[cols])
                        churn_count = sum(pred)
                        return f"Customer churn predictions: {churn_count} out of {len(pred)} customers likely to churn (Predictions: {list(pred[:5])})"
                    else:
                        return "Missing required columns for churn model."
                except Exception as e:
                    return f"Error loading churn model: {str(e)}"
            else:
                return "Please upload churn/customer data in the 'Churn/Supplier' tab first."
        
        # Module 3: Supplier Reliability
        elif "reliability" in user_lower or "supplier" in user_lower:
            df = st.session_state["uploaded_data"].get("churn")
            if df is not None:
                try:
                    model_supplier = joblib.load("catboost_supplier_reliability.pkl")
                    cols = ['Order Item Quantity', 'Order Profit Per Order', 'Sales']
                    if all(col in df.columns for col in cols):
                        pred = model_supplier.predict(df[cols])
                        return f"Supplier reliability scores: {list(pred[:5])} (showing first 5)"
                    else:
                        return "Missing required columns for supplier reliability model."
                except Exception as e:
                    return f"Error loading supplier model: {str(e)}"
            else:
                return "Please upload supplier data in the 'Churn/Supplier' tab first."
        
        else:
            return None
    
    if user_input:
        # Get project model response
        project_response = call_project_models(user_input)
        
        if project_response:
            # Build prompt for Gemini
            prompt = f"""You are a supply chain AI assistant. A user asked: "{user_input}"
            
Here is the ML model prediction/analysis result: {project_response}

Please provide a clear, concise, and helpful explanation of this result in 2-3 sentences. 
Make it easy to understand for a business user."""
            
            # Call Gemini API with error handling
            try:
                if model:
                    gemini_response = model.generate_content(prompt).text.strip()
                else:
                    gemini_response = project_response + "\n\n(Gemini API not configured, showing raw model output above.)"
            except Exception as e:
                gemini_response = f"{project_response}\n\n⚠️ Gemini explanation unavailable: {str(e)}"
        else:
            gemini_response = "I'm your supply chain AI assistant. I can help with:\n- Late delivery risk predictions\n- Delay forecasting\n- Demand forecasts\n- Customer churn analysis\n- Supplier reliability scoring\n\nPlease ask a specific question or upload data for the relevant module."
        
        # Add to chat history
        st.session_state['chat_history'].append({"role": "user", "content": user_input})
        st.session_state['chat_history'].append({"role": "assistant", "content": gemini_response})
        
        # Display latest response
        st.write("**AI Response:**")
        st.write(gemini_response)
    
    # Display chat history
    if st.session_state['chat_history']:
        st.subheader("📝 Chat History")
        for entry in st.session_state['chat_history']:
            role_icon = "👤" if entry["role"] == "user" else "🤖"
            st.markdown(f"**{role_icon} {entry['role'].upper()}:** {entry['content']}")
