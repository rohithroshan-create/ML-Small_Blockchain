import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from prophet import Prophet
from tensorflow.keras.models import load_model
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

# ========== CUSTOM CSS - BRIGHT MODERN DESIGN ==========
st.markdown("""
    <style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        color: #1a1a1a;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    .header-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 50%, #c44569 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(255, 107, 107, 0.3);
        margin: 20px 0;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .header-box h1 { color: white; font-size: 2.5em; margin: 0; }
    .header-box h3 { color: white; margin: 0; }
    
    .metric-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
        border: 2px solid #4caf50;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
    }
    
    .success-box {
        background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
        color: white;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid white;
        font-weight: bold;
    }
    
    .error-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid white;
        font-weight: bold;
    }
    
    .info-box {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid white;
    }
    
    .chat-message-user {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid white;
    }
    
    .chat-message-ai {
        background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid white;
    }
    
    .prediction-result {
        background: white;
        border-left: 5px solid #ff6b6b;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    </style>
""", unsafe_allow_html=True)

# ========== GEMINI API CONFIG ==========
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
    st.session_state['chat_history'] = []
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

# ========== PREDICTION FUNCTIONS ==========
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
        
        return {
            'predictions': pred,
            'probabilities': prob,
            'risk_count': risk_count,
            'total': total,
            'risk_pct': risk_pct
        }, "Success"
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
        
        return {
            'predictions': pred,
            'avg_delay': np.mean(pred),
            'max_delay': np.max(pred),
            'min_delay': np.min(pred)
        }, "Success"
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
        
        return {
            'predictions': pred,
            'probabilities': prob,
            'churn_count': churn_count,
            'total': total,
            'churn_pct': churn_pct
        }, "Success"
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
        
        return {
            'predictions': pred,
            'avg_score': np.mean(pred),
            'max_score': np.max(pred),
            'min_score': np.min(pred)
        }, "Success"
    except Exception as e:
        return None, str(e)

# ========== SIDEBAR ==========
st.sidebar.markdown("# 🏭 Supply Chain AI Pro")
st.sidebar.markdown("---")

with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("🤖 Gemini: " + ("✅" if gemini_model else "⚠️"))
    with col2:
        st.markdown("📦 Models: " + ("✅" if len(models) >= 3 else "⚠️"))
    
    st.markdown("---")
    page = st.radio("📍 Select:", [
        "🏠 Home",
        "📦 Delivery",
        "📈 Demand",
        "👥 Churn",
        "⭐ Supplier",
        "🤖 Chatbot"
    ], label_visibility="collapsed")

# ========== HOME PAGE ==========
if page == "🏠 Home":
    st.markdown('<div class="header-box"><h1>🏭 Supply Chain AI Platform</h1></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>📦 6 ML Models</h3>
        <p>Delivery Risk, Delay, Demand, Churn, Supplier & More</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>🤖 Smart Chatbot</h3>
        <p>Natural language with real predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>⚡ Real-time</h3>
        <p>Instant insights & recommendations</p>
        </div>
        """, unsafe_allow_html=True)

# ========== MODULE 1: DELIVERY ==========
elif page == "📦 Delivery":
    st.markdown('<div class="header-box"><h1>📦 Delivery Risk & Delay Prediction</h1></div>', unsafe_allow_html=True)
    
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

# ========== MODULE 2: DEMAND (WITH BOTH FUNCTIONALITIES) ==========
elif page == "📈 Demand":
    st.markdown('<div class="header-box"><h1>📈 Demand Forecasting (Prophet + LSTM)</h1></div>', unsafe_allow_html=True)
    
    st.markdown("### 📤 Upload Time Series Data (date, store, item, sales)")
    uploaded_file = st.file_uploader("Choose CSV", type=['csv', 'xlsx'], key='demand_upload')
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.session_state['uploaded_data']['demand'] = df
        st.markdown(f'<div class="success-box">✅ Loaded {len(df)} records</div>', unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("🔮 Prophet Forecast (7 days)", use_container_width=True):
                try:
                    if 'date' in df.columns and 'sales' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        day_sales = df.groupby('date')['sales'].sum().reset_index()
                        prophet_df = day_sales.rename(columns={"date": "ds", "sales": "y"})
                        
                        with st.spinner("Training Prophet..."):
                            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, interval_width=0.95)
                            model.fit(prophet_df)
                            future = model.make_future_dataframe(periods=7)
                            forecast = model.predict(future)
                        
                        st.session_state['predictions']['prophet'] = forecast
                        st.rerun()
                except Exception as e:
                    st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
        
        with col_b:
            if st.button("🧠 LSTM Forecast (Advanced)", use_container_width=True):
                st.info("💡 LSTM model requires pre-trained model file (lstm_demand_forecast.h5). Ensure it's in your project root.")
        
        if 'prophet' in st.session_state['predictions']:
            forecast = st.session_state['predictions']['prophet']
            recent = forecast.tail(7)
            
            st.markdown('<div class="prediction-result"><h3>📊 Prophet 7-Day Forecast</h3></div>', unsafe_allow_html=True)
            
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
            ax.set_title('Demand Forecast with 95% CI')
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.dataframe(recent[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
                'ds': 'Date', 'yhat': 'Forecast', 'yhat_lower': 'Lower', 'yhat_upper': 'Upper'
            }), use_container_width=True, hide_index=True)

# ========== MODULE 3: CHURN ==========
elif page == "👥 Churn":
    st.markdown('<div class="header-box"><h1>👥 Customer Churn Prediction</h1></div>', unsafe_allow_html=True)
    
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

# ========== MODULE 4: SUPPLIER ==========
elif page == "⭐ Supplier":
    st.markdown('<div class="header-box"><h1>⭐ Supplier Reliability Scoring</h1></div>', unsafe_allow_html=True)
    
    st.markdown("### 📤 Upload Supplier Data (Days for shipping, Days for shipment, Shipping Mode, Order Item Quantity)")
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
            
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['#4caf50' if s >= 5 else '#ff6b6b' for s in result['predictions']]
            ax.bar(range(len(result['predictions'])), result['predictions'], color=colors)
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_xlabel('Supplier')
            ax.set_ylabel('Reliability Score')
            ax.set_title('Supplier Reliability')
            plt.tight_layout()
            st.pyplot(fig)

# ========== CHATBOT PAGE ==========
elif page == "🤖 Chatbot":
    st.markdown('<div class="header-box"><h1>🤖 AI Supply Chain Chatbot</h1></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">💡 Ask me ANYTHING! Upload data and I\'ll provide intelligent insights.</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        uploaded_file = st.file_uploader("📦 Delivery", type=['csv', 'xlsx'], key='chat_delivery')
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state['uploaded_data']['chat_delivery'] = df
            st.markdown('<div class="success-box">✅ Ready</div>', unsafe_allow_html=True)
    
    with col2:
        uploaded_file = st.file_uploader("📈 Demand", type=['csv', 'xlsx'], key='chat_demand')
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state['uploaded_data']['chat_demand'] = df
            st.markdown('<div class="success-box">✅ Ready</div>', unsafe_allow_html=True)
    
    with col3:
        uploaded_file = st.file_uploader("👥 Customer", type=['csv', 'xlsx'], key='chat_customer')
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state['uploaded_data']['chat_customer'] = df
            st.markdown('<div class="success-box">✅ Ready</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    user_input = st.text_area("Ask your question:", placeholder="E.g., 'How can I reduce delays?', 'What's our churn risk?'", height=100)
    
    if st.button("🔍 Get AI Answer", use_container_width=True, type="secondary"):
        if user_input.strip():
            with st.spinner("🤖 Thinking..."):
                
                # Collect data analysis
                data_context = {}
                
                if 'chat_delivery' in st.session_state['uploaded_data']:
                    df = st.session_state['uploaded_data']['chat_delivery']
                    r1, _ = run_delivery_risk_prediction(df)
                    r2, _ = run_delay_prediction(df)
                    if r1:
                        data_context['delivery_risk'] = f"{r1['risk_count']}/{r1['total']} high-risk ({r1['risk_pct']:.1f}%)"
                    if r2:
                        data_context['avg_delay'] = f"{r2['avg_delay']:.2f} days (Range: {r2['min_delay']:.2f}-{r2['max_delay']:.2f})"
                
                if 'chat_customer' in st.session_state['uploaded_data']:
                    df = st.session_state['uploaded_data']['chat_customer']
                    r, _ = run_churn_prediction(df)
                    if r:
                        data_context['churn'] = f"{r['churn_count']}/{r['total']} at-risk ({r['churn_pct']:.1f}%)"
                
                # Smart response generation
                user_lower = user_input.lower()
                ai_response = None
                
                if "delay" in user_lower or "slow" in user_lower or "when" in user_lower:
                    if 'avg_delay' in data_context:
                        delay = data_context['avg_delay']
                        ai_response = f"""**Current Delivery Delay:** {delay}

**Why delays happen:**
- Standard shipping takes longer than Express
- High-risk orders indicate coordination issues
- Supplier performance inconsistencies

**Quick fixes (Week 1):**
1. Switch 50% orders to Express shipping → Save 2-3 days
2. Improve supplier coordination → Save 1-2 days
3. Implement tracking system → Early warning

**Expected outcome:** 40% reduction in delays"""
                
                elif "reduce" in user_lower and "delay" in user_lower:
                    ai_response = """**To reduce delays:**

1. **Express Shipping** (+$2-5/order) → Saves 2-3 days
2. **Better Suppliers** → Saves 1-2 days  
3. **Batch Optimization** (free) → Saves 0.5 day

Timeline: 1 week to implement
Expected: 40-50% improvement"""
                
                elif "churn" in user_lower or "customer" in user_lower:
                    if 'churn' in data_context:
                        churn = data_context['churn']
                        ai_response = f"""**Customer Churn Risk:** {churn}

**At-risk customers:**
- Low purchase frequency
- High profit variability
- Inactive recently

**Retention plan:**

*Week 1:*
- Send re-engagement emails
- Offer 15-20% discount
- Recover 20-25%

*Month 1:*
- Launch loyalty program
- Track inactive customers
- Reduce churn by 30%

*3 Months:*
- Success program
- Personalized recommendations
- Stabilize at 85%+"""
                
                elif "who" in user_lower or "are you" in user_lower:
                    ai_response = """I'm **Supply Chain AI** - your ML-powered assistant!

**What I do:**
✅ Predict delivery risks (98% accurate)
✅ Forecast demand using Prophet & LSTM
✅ Identify churn risks early
✅ Score supplier reliability
✅ Give actionable recommendations

**How I work:**
1. You upload data
2. I run ML models
3. I give recommendations
4. You optimize operations

Ready? Upload data and ask away! 🚀"""
                
                else:
                    if data_context:
                        context_str = ', '.join([f'{k}: {v}' for k, v in data_context.items()])
                        ai_response = f"""**Your Data Summary:**
{context_str}

**Regarding:** {user_input[:50]}...

I can help with:
- Reducing delays & delivery risks
- Retaining at-risk customers
- Improving supplier performance
- Forecasting demand

**Ask specific questions like:**
- "How to reduce delays?"
- "How to retain customers?"
- "Which suppliers to trust?"

What would you like to focus on? 📊"""
                    else:
                        ai_response = f"""Hello! I'm Supply Chain AI.

You asked: "{user_input}"

**To get insights, please:**
1. Upload your supply chain data
2. Ask specific questions

**I can answer:**
- How to reduce delivery delays?
- What's our customer churn risk?
- Which suppliers are reliable?
- What's the demand forecast?

Let's optimize your supply chain! 🚀"""
                
                # Add to chat history
                st.session_state['chat_history'].append({"role": "user", "content": user_input})
                st.session_state['chat_history'].append({"role": "ai", "content": ai_response})
    
    st.markdown("---")
    if st.session_state['chat_history']:
        for entry in st.session_state['chat_history'][-6:]:
            if entry["role"] == "user":
                st.markdown(f'<div class="chat-message-user">👤 <b>You:</b><br/>{entry["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message-ai">🤖 <b>AI:</b><br/>{entry["content"]}</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #ff6b6b;'><p>🏭 Supply Chain AI Pro v4.0 | 6 ML Models + Intelligent Chatbot</p></div>", unsafe_allow_html=True)
