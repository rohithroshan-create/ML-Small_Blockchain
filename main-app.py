import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from prophet import Prophet
from sklearn.preprocessing import LabelEncoder
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
    
    .section-divider {
        border: 2px solid #ff6b6b;
        margin: 30px 0;
        border-radius: 5px;
    }
    
    .metric-box {
        background: white;
        border: 2px solid #ff6b6b;
        padding: 20px;
        border-radius: 12px;
        margin: 10px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .prediction-result {
        background: white;
        border-left: 5px solid #ff6b6b;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .button-primary {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
    }
    
    </style>
""", unsafe_allow_html=True)

# ========== GEMINI API CONFIG ==========
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
gemini_model = None
gemini_connected = False

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        gemini_connected = True
    except:
        gemini_connected = False

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
    except Exception as e:
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
            return None, f"Missing columns. Required: {cols}"
        
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
            return None, f"Missing columns. Required: {cols}"
        
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
            return None, f"Missing columns. Required: {cols}"
        
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
            return None, f"Missing columns. Required: {cols}"
        
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

def generate_intelligent_response(user_input, data_analysis):
    """Generate intelligent conversational response"""
    if not gemini_model:
        return f"Data Analysis: {data_analysis}"
    
    try:
        prompt = f"""You are an expert supply chain consultant. Answer this question conversationally like ChatGPT but use the provided data:

Question: {user_input}

Data Analysis: {data_analysis}

Provide a natural, conversational answer that:
1. Directly addresses the question
2. Uses data to support your answer
3. Gives 2-3 specific recommendations
4. Be helpful and friendly"""
        
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.8, top_p=0.95, max_output_tokens=800)
        )
        return response.text
    except:
        return f"Analysis: {data_analysis}"

# ========== SIDEBAR ==========
st.sidebar.markdown("# 🏭 Supply Chain AI Pro")
st.sidebar.markdown("---")

with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("🤖 Gemini: " + ("✅" if gemini_connected else "❌"))
    with col2:
        st.markdown("📦 Models: " + ("✅" if models else "❌"))
    
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
        <div class="metric-box">
        <h3>📦 6 ML Models</h3>
        <p>Delivery Risk, Delay Forecast, Churn Prediction, Supplier Reliability & More</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
        <h3>🤖 AI Chatbot</h3>
        <p>Natural language queries with real model insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
        <h3>⚡ Real-time</h3>
        <p>Instant predictions and recommendations</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    ### 🎯 Features
    - **Delivery Risk Assessment** - Identify at-risk shipments
    - **Delay Forecasting** - Predict delivery delays
    - **Demand Forecasting** - Prophet + LSTM time series
    - **Customer Churn** - Retain valuable customers
    - **Supplier Reliability** - Score supplier performance
    - **AI Chatbot** - Ask anything about supply chain
    """)

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
            
            with st.expander("📋 Preview Data"):
                st.dataframe(df.head(3), use_container_width=True)
            
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
            
            st.markdown('<div class="prediction-result"><h3>🎯 Late Delivery Risk</h3></div>', unsafe_allow_html=True)
            
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
            
            st.markdown('<div class="prediction-result"><h3>⏱️ Delay Forecast</h3></div>', unsafe_allow_html=True)
            
            col_i, col_ii, col_iii = st.columns(3)
            col_i.metric("📊 Avg", f"{result['avg_delay']:.2f}", "days")
            col_ii.metric("⬆️ Max", f"{result['max_delay']:.2f}", "days")
            col_iii.metric("⬇️ Min", f"{result['min_delay']:.2f}", "days")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(result['predictions'], bins=15, color='#ff6b6b', edgecolor='white', linewidth=1.5)
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_xlabel('Delay (Days)', color='#1a1a1a')
            ax.set_ylabel('Frequency', color='#1a1a1a')
            ax.set_title('Delivery Delay Distribution', color='#1a1a1a', fontsize=14, fontweight='bold')
            ax.tick_params(colors='#1a1a1a')
            plt.tight_layout()
            st.pyplot(fig)

# ========== MODULE 2: DEMAND ==========
elif page == "📈 Demand":
    st.markdown('<div class="header-box"><h1>📈 Demand & Inventory Forecasting</h1></div>', unsafe_allow_html=True)
    
    st.markdown("### 📤 Upload Time Series Data (date, store, item, sales)")
    uploaded_file = st.file_uploader("Choose CSV", type=['csv', 'xlsx'], key='demand_upload')
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.session_state['uploaded_data']['demand'] = df
        st.markdown(f'<div class="success-box">✅ Loaded {len(df)} records</div>', unsafe_allow_html=True)
        
        if st.button("🔮 Forecast (7 days)", use_container_width=True):
            try:
                if 'date' in df.columns and 'sales' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    day_sales = df.groupby('date')['sales'].sum().reset_index()
                    prophet_df = day_sales.rename(columns={"date": "ds", "sales": "y"})
                    
                    with st.spinner("Training..."):
                        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, interval_width=0.95)
                        model.fit(prophet_df)
                        future = model.make_future_dataframe(periods=7)
                        forecast = model.predict(future)
                    
                    st.session_state['predictions']['prophet'] = forecast
                    st.rerun()
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
        
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
            ax.set_xlabel('Date', color='#1a1a1a')
            ax.set_ylabel('Demand', color='#1a1a1a')
            ax.set_title('Demand Forecast (95% CI)', color='#1a1a1a', fontsize=14, fontweight='bold')
            ax.legend(facecolor='#f8f9fa', edgecolor='#1a1a1a')
            ax.tick_params(colors='#1a1a1a')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

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
            col_i.metric("📊 Avg Score", f"{result['avg_score']:.2f}", "/10")
            col_ii.metric("⬆️ Best", f"{result['max_score']:.2f}", "/10")
            col_iii.metric("⬇️ Worst", f"{result['min_score']:.2f}", "/10")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['#4caf50' if s >= 5 else '#ff6b6b' for s in result['predictions']]
            ax.bar(range(len(result['predictions'])), result['predictions'], color=colors, edgecolor='#1a1a1a', linewidth=1.5)
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_xlabel('Supplier', color='#1a1a1a')
            ax.set_ylabel('Reliability Score', color='#1a1a1a')
            ax.set_title('Supplier Reliability Scores', color='#1a1a1a', fontsize=14, fontweight='bold')
            ax.axhline(y=5, color='orange', linestyle='--', label='Threshold')
            ax.tick_params(colors='#1a1a1a')
            ax.legend(facecolor='#f8f9fa', edgecolor='#1a1a1a')
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
            with st.spinner("🤖 Analyzing..."):
                analysis = []
                
                if 'chat_delivery' in st.session_state['uploaded_data']:
                    df = st.session_state['uploaded_data']['chat_delivery']
                    r1, _ = run_delivery_risk_prediction(df)
                    r2, _ = run_delay_prediction(df)
                    if r1:
                        analysis.append(f"Delivery: {r1['risk_count']}/{r1['total']} high-risk ({r1['risk_pct']:.1f}%)")
                    if r2:
                        analysis.append(f"Avg delay: {r2['avg_delay']:.2f} days")
                
                if 'chat_demand' in st.session_state['uploaded_data']:
                    analysis.append("Demand data uploaded")
                
                if 'chat_customer' in st.session_state['uploaded_data']:
                    df = st.session_state['uploaded_data']['chat_customer']
                    r, _ = run_churn_prediction(df)
                    if r:
                        analysis.append(f"Churn risk: {r['churn_count']}/{r['total']} ({r['churn_pct']:.1f}%)")
                
                context = " | ".join(analysis) if analysis else "No data analyzed"
                response = generate_intelligent_response(user_input, context)
                
                st.session_state['chat_history'].append({"role": "user", "content": user_input})
                st.session_state['chat_history'].append({"role": "ai", "content": response})
    
    st.markdown("---")
    if st.session_state['chat_history']:
        for entry in st.session_state['chat_history'][-6:]:
            if entry["role"] == "user":
                st.markdown(f'<div class="chat-message-user">👤 <b>You:</b><br/>{entry["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message-ai">🤖 <b>AI:</b><br/>{entry["content"]}</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #ff6b6b; margin-top: 50px;'><p>🏭 Supply Chain AI Pro v3.0 | Built with CatBoost, Prophet & Gemini</p></div>", unsafe_allow_html=True)
