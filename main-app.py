import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import warnings
warnings.filterwarnings('ignore')

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="🏭 Supply Chain AI Pro",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
st.markdown("""
    <style>
    * { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .main { background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); color: #1a1a1a; }
    .stApp { background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); }
    .header-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 50%, #c44569 100%);
        color: white; padding: 30px; border-radius: 15px;
        box-shadow: 0 8px 32px rgba(255, 107, 107, 0.3);
        margin: 20px 0; border: 1px solid rgba(255,255,255,0.2);
    }
    .header-box h1 { color: white; font-size: 2.5em; margin: 0; }
    .metric-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
        border: 2px solid #4caf50; padding: 20px; border-radius: 12px;
        margin: 10px 0; box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
    }
    .success-box {
        background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
        color: white; padding: 12px; border-radius: 8px; margin: 10px 0;
        border-left: 4px solid white; font-weight: bold;
    }
    .error-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white; padding: 12px; border-radius: 8px; margin: 10px 0;
        border-left: 4px solid white; font-weight: bold;
    }
    .info-box {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        color: white; padding: 15px; border-radius: 10px; margin: 10px 0;
        border-left: 4px solid white;
    }
    .chat-message-user {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        color: white; padding: 15px; border-radius: 10px; margin: 10px 0;
        border-left: 4px solid white;
    }
    .chat-message-ai {
        background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
        color: white; padding: 15px; border-radius: 10px; margin: 10px 0;
        border-left: 4px solid white;
    }
    .prediction-result {
        background: white; border-left: 5px solid #ff6b6b; padding: 20px;
        border-radius: 10px; margin: 15px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ========== GEMINI CONFIG ==========
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    except:
        pass

# ========== SESSION STATE ==========
if 'uploaded_data' not in st.session_state:
    st.session_state['uploaded_data'] = {}
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = {}
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = {}

# ========== MODEL LOADING ==========
@st.cache_resource
def load_models():
    models = {}
    try:
        if os.path.exists("catboost_delivery_risk.pkl"):
            models['delivery_risk'] = joblib.load("catboost_delivery_risk.pkl")
        if os.path.exists("catboost_delay_regression.pkl"):
            models['delay_regression'] = joblib.load("catboost_delay_regression.pkl")
        if os.path.exists("catboost_customer_churn.pkl"):
            models['churn'] = joblib.load("catboost_customer_churn.pkl")
        if os.path.exists("catboost_supplier_reliability.pkl"):
            models['supplier'] = joblib.load("catboost_supplier_reliability.pkl")
    except:
        pass
    return models

models = load_models()

# ========== UTILITY FUNCTIONS ==========
def get_gemini_response(user_question, context=""):
    """Get TRUE AI response - not restricted by predictions"""
    if not gemini_model:
        return None
    
    try:
        prompt = f"""You are an expert supply chain consultant. Answer this question naturally and helpfully.

User Question: {user_question}

Context (if available): {context}

Provide a helpful, conversational answer. If the user asks for predictions, mention the data if available.
If they ask general questions, answer from your expertise. Be practical and actionable."""
        
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.8, top_p=0.95, max_output_tokens=1000)
        )
        return response.text
    except:
        return None

def run_delivery_risk_prediction(df):
    try:
        if 'delivery_risk' not in models:
            return None, "Model not found"
        cols = ['Days for shipping (real)', 'Days for shipment (scheduled)', 'Shipping Mode', 'Order Item Quantity']
        if not all(col in df.columns for col in cols):
            return None, f"Missing columns"
        model = models['delivery_risk']
        pred = model.predict(df[cols])
        prob = model.predict_proba(df[cols])
        risk_count = sum(pred)
        total = len(pred)
        risk_pct = (risk_count / total) * 100
        return {'predictions': pred, 'probabilities': prob, 'risk_count': risk_count, 'total': total, 'risk_pct': risk_pct}, "Success"
    except Exception as e:
        return None, str(e)

def run_delay_prediction(df):
    try:
        if 'delay_regression' not in models:
            return None, "Model not found"
        cols = ['Days for shipping (real)', 'Days for shipment (scheduled)', 'Shipping Mode', 'Order Item Quantity']
        if not all(col in df.columns for col in cols):
            return None, f"Missing columns"
        model = models['delay_regression']
        pred = model.predict(df[cols])
        return {'predictions': pred, 'avg_delay': np.mean(pred), 'max_delay': np.max(pred), 'min_delay': np.min(pred)}, "Success"
    except Exception as e:
        return None, str(e)

def run_churn_prediction(df):
    try:
        if 'churn' not in models:
            return None, "Model not found"
        cols = ['Customer Segment', 'Type', 'Category Name', 'Order Item Quantity', 'Sales', 'Order Profit Per Order']
        if not all(col in df.columns for col in cols):
            return None, f"Missing columns"
        model = models['churn']
        pred = model.predict(df[cols])
        prob = model.predict_proba(df[cols])
        churn_count = sum(pred)
        total = len(pred)
        churn_pct = (churn_count / total) * 100
        return {'predictions': pred, 'probabilities': prob, 'churn_count': churn_count, 'total': total, 'churn_pct': churn_pct}, "Success"
    except Exception as e:
        return None, str(e)

def run_supplier_prediction(df):
    try:
        if 'supplier' not in models:
            return None, "Model not found"
        cols = ['Days for shipping (real)', 'Days for shipment (scheduled)', 'Shipping Mode', 'Order Item Quantity']
        if not all(col in df.columns for col in cols):
            return None, f"Missing columns"
        model = models['supplier']
        pred = model.predict(df[cols])
        return {'predictions': pred, 'avg_score': np.mean(pred), 'max_score': np.max(pred), 'min_score': np.min(pred)}, "Success"
    except Exception as e:
        return None, str(e)

def embedded_chatbot(module_name, df=None):
    """Embedded chatbot for any module"""
    st.markdown("---")
    st.markdown("### 💬 AI Assistant for this Module")
    
    user_input = st.text_input(f"Ask anything about {module_name}:", key=f"chat_{module_name}")
    
    if user_input:
        with st.spinner("🤖 Thinking..."):
            # Create context from current data
            context = ""
            if df is not None:
                context = f"Current data has {len(df)} records with columns: {', '.join(df.columns[:5])}"
            
            # Get AI response
            ai_response = get_gemini_response(user_input, context)
            
            if not ai_response:
                # Fallback intelligent responses
                ai_response = f"Based on your question '{user_input}', here's what I can help with in {module_name}:\n\n"
                if "prediction" in user_input.lower():
                    ai_response += "I can run predictions on your data. Upload a CSV and I'll analyze it for you."
                elif "how to" in user_input.lower():
                    ai_response += "I provide actionable recommendations based on supply chain best practices."
                elif "why" in user_input.lower():
                    ai_response += "Let me help you understand the supply chain dynamics of your data."
                else:
                    ai_response += "I'm here to help! Ask me anything about supply chain optimization, predictions, or strategies."
            
            # Display conversation
            if module_name not in st.session_state['chat_history']:
                st.session_state['chat_history'][module_name] = []
            
            st.session_state['chat_history'][module_name].append({"role": "user", "content": user_input})
            st.session_state['chat_history'][module_name].append({"role": "ai", "content": ai_response})
            
            st.markdown(f'<div class="chat-message-user">👤 <b>You:</b><br/>{user_input}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-message-ai">🤖 <b>AI:</b><br/>{ai_response}</div>', unsafe_allow_html=True)

# ========== SIDEBAR ==========
st.sidebar.markdown("# 🏭 Supply Chain AI Pro")
st.sidebar.markdown("---")

with st.sidebar:
    page = st.radio("📍 Select:", ["🏠 Home", "📦 Delivery", "📈 Demand", "👥 Churn", "⭐ Supplier"], label_visibility="collapsed")

# ========== HOME PAGE ==========
if page == "🏠 Home":
    st.markdown('<div class="header-box"><h1>🏭 Supply Chain AI Platform</h1></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h3>📦 6 ML Models</h3><p>All integrated with AI</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>🤖 Smart Chatbot</h3><p>In every module</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>⚡ Real-time</h3><p>Instant insights</p></div>', unsafe_allow_html=True)

# ========== MODULE 1: DELIVERY ==========
elif page == "📦 Delivery":
    st.markdown('<div class="header-box"><h1>📦 Delivery Risk & Delay</h1></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("### 📤 Upload Data")
        uploaded_file = st.file_uploader("Choose CSV", type=['csv', 'xlsx'], key='delivery_upload')
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state['uploaded_data']['delivery'] = df
            st.markdown(f'<div class="success-box">✅ Loaded {len(df)} records</div>', unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔴 Risk", use_container_width=True):
                    result, msg = run_delivery_risk_prediction(df)
                    if result:
                        st.session_state['predictions']['delivery_risk'] = result
                        st.rerun()
                    else:
                        st.markdown(f'<div class="error-box">❌ {msg}</div>', unsafe_allow_html=True)
            with col_b:
                if st.button("⏱️ Delay", use_container_width=True):
                    result, msg = run_delay_prediction(df)
                    if result:
                        st.session_state['predictions']['delay'] = result
                        st.rerun()
                    else:
                        st.markdown(f'<div class="error-box">❌ {msg}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Results")
        
        if 'delivery_risk' in st.session_state['predictions']:
            result = st.session_state['predictions']['delivery_risk']
            col_i, col_ii, col_iii = st.columns(3)
            col_i.metric("🔴 High Risk", result['risk_count'], f"{result['risk_pct']:.1f}%")
            col_ii.metric("🟢 On-Time", result['total'] - result['risk_count'], f"{100-result['risk_pct']:.1f}%")
            col_iii.metric("📦 Total", result['total'], "100%")
            
            st.dataframe({
                'Order': range(len(result['predictions'])),
                'Status': ['🔴 HIGH RISK' if p == 1 else '🟢 ON TIME' for p in result['predictions']],
                'Confidence': [f"{max(result['probabilities'][i])*100:.1f}%" for i in range(len(result['predictions']))]
            }, use_container_width=True, hide_index=True)
        
        if 'delay' in st.session_state['predictions']:
            result = st.session_state['predictions']['delay']
            col_i, col_ii, col_iii = st.columns(3)
            col_i.metric("📊 Avg", f"{result['avg_delay']:.2f}", "days")
            col_ii.metric("⬆️ Max", f"{result['max_delay']:.2f}", "days")
            col_iii.metric("⬇️ Min", f"{result['min_delay']:.2f}", "days")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(result['predictions'], bins=15, color='#ff6b6b', edgecolor='white', linewidth=1.5)
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_xlabel('Delay (Days)')
            ax.set_ylabel('Frequency')
            ax.set_title('Delivery Delay Distribution')
            plt.tight_layout()
            st.pyplot(fig)
    
    # EMBEDDED CHATBOT
    df = st.session_state['uploaded_data'].get('delivery')
    embedded_chatbot("Delivery Module", df)

# ========== MODULE 2: DEMAND ==========
elif page == "📈 Demand":
    st.markdown('<div class="header-box"><h1>📈 Demand Forecasting</h1></div>', unsafe_allow_html=True)
    
    st.markdown("### 📤 Upload Time Series Data (date, sales)")
    uploaded_file = st.file_uploader("Choose CSV", type=['csv', 'xlsx'], key='demand_upload')
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.session_state['uploaded_data']['demand'] = df
        st.markdown(f'<div class="success-box">✅ Loaded {len(df)} records</div>', unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("🔮 Prophet Forecast", use_container_width=True):
                try:
                    if 'date' in df.columns and 'sales' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        day_sales = df.groupby('date')['sales'].sum().reset_index()
                        prophet_df = day_sales.rename(columns={"date": "ds", "sales": "y"})
                        with st.spinner("Training..."):
                            model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                            model.fit(prophet_df)
                            future = model.make_future_dataframe(periods=7)
                            forecast = model.predict(future)
                        st.session_state['predictions']['prophet'] = forecast
                        st.rerun()
                except Exception as e:
                    st.markdown(f'<div class="error-box">❌ {str(e)}</div>', unsafe_allow_html=True)
        
        with col_b:
            st.info("💡 LSTM model ready when you need it!")
        
        if 'prophet' in st.session_state['predictions']:
            forecast = st.session_state['predictions']['prophet']
            recent = forecast.tail(7)
            
            col_i, col_ii, col_iii = st.columns(3)
            col_i.metric("📈 Avg", f"{recent['yhat'].mean():.0f}", "units")
            col_ii.metric("⬆️ Peak", f"{recent['yhat'].max():.0f}", "units")
            col_iii.metric("Range", f"±{recent['yhat_upper'].mean() - recent['yhat'].mean():.0f}", "units")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='#ff6b6b', linewidth=2.5)
            ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2, color='#ff6b6b')
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_xlabel('Date')
            ax.set_ylabel('Demand')
            ax.set_title('7-Day Demand Forecast')
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
    
    # EMBEDDED CHATBOT
    df = st.session_state['uploaded_data'].get('demand')
    embedded_chatbot("Demand Module", df)

# ========== MODULE 3: CHURN ==========
elif page == "👥 Churn":
    st.markdown('<div class="header-box"><h1>👥 Customer Churn</h1></div>', unsafe_allow_html=True)
    
    st.markdown("### 📤 Upload Customer Data")
    uploaded_file = st.file_uploader("Choose CSV", type=['csv', 'xlsx'], key='churn_upload')
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.session_state['uploaded_data']['churn'] = df
        st.markdown(f'<div class="success-box">✅ Loaded {len(df)} records</div>', unsafe_allow_html=True)
        
        if st.button("🎯 Predict Churn", use_container_width=True):
            result, msg = run_churn_prediction(df)
            if result:
                st.session_state['predictions']['churn'] = result
                st.rerun()
            else:
                st.markdown(f'<div class="error-box">❌ {msg}</div>', unsafe_allow_html=True)
        
        if 'churn' in st.session_state['predictions']:
            result = st.session_state['predictions']['churn']
            col_i, col_ii, col_iii = st.columns(3)
            col_i.metric("🔴 At-Risk", result['churn_count'], f"{result['churn_pct']:.1f}%")
            col_ii.metric("🟢 Retained", result['total'] - result['churn_count'], f"{100-result['churn_pct']:.1f}%")
            col_iii.metric("👥 Total", result['total'], "100%")
    
    # EMBEDDED CHATBOT
    df = st.session_state['uploaded_data'].get('churn')
    embedded_chatbot("Churn Module", df)

# ========== MODULE 4: SUPPLIER ==========
elif page == "⭐ Supplier":
    st.markdown('<div class="header-box"><h1>⭐ Supplier Reliability</h1></div>', unsafe_allow_html=True)
    
    st.markdown("### 📤 Upload Supplier Data")
    uploaded_file = st.file_uploader("Choose CSV", type=['csv', 'xlsx'], key='supplier_upload')
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.session_state['uploaded_data']['supplier'] = df
        st.markdown(f'<div class="success-box">✅ Loaded {len(df)} records</div>', unsafe_allow_html=True)
        
        if st.button("⭐ Score Reliability", use_container_width=True):
            result, msg = run_supplier_prediction(df)
            if result:
                st.session_state['predictions']['supplier'] = result
                st.rerun()
            else:
                st.markdown(f'<div class="error-box">❌ {msg}</div>', unsafe_allow_html=True)
        
        if 'supplier' in st.session_state['predictions']:
            result = st.session_state['predictions']['supplier']
            col_i, col_ii, col_iii = st.columns(3)
            col_i.metric("📊 Avg", f"{result['avg_score']:.2f}", "/10")
            col_ii.metric("⬆️ Best", f"{result['max_score']:.2f}", "/10")
            col_iii.metric("⬇️ Worst", f"{result['min_score']:.2f}", "/10")
    
    # EMBEDDED CHATBOT
    df = st.session_state['uploaded_data'].get('supplier')
    embedded_chatbot("Supplier Module", df)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #ff6b6b;'><p>🏭 Supply Chain AI Pro v5.0 | Embedded Chatbot + Smart AI</p></div>", unsafe_allow_html=True)
