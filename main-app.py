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

# ========== CUSTOM CSS - ADVANCED STYLING ==========
st.markdown("""
    <style>
    :root {
        --primary: #667eea;
        --secondary: #764ba2;
        --success: #34d399;
        --danger: #f87171;
        --warning: #fbbf24;
    }
    
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    .header-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        margin: 20px 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px solid #667eea;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .prediction-high-risk {
        background: linear-gradient(135deg, #f87171 0%, #dc2626 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .prediction-safe {
        background: linear-gradient(135deg, #34d399 0%, #059669 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .section-title {
        color: #667eea;
        font-size: 2em;
        font-weight: bold;
        margin: 30px 0 15px 0;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
    }
    
    .chat-message-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    
    .chat-message-ai {
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #34d399;
    }
    
    .error-box {
        background: linear-gradient(135deg, #f87171 0%, #dc2626 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #f87171;
    }
    
    .success-box {
        background: linear-gradient(135deg, #34d399 0%, #059669 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #34d399;
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
if 'model_outputs' not in st.session_state:
    st.session_state['model_outputs'] = {}

# ========== UTILITY FUNCTIONS ==========
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
        st.error(f"Error loading models: {str(e)}")
    return models

models = load_models()

def run_delivery_risk_prediction(df):
    """Run delivery risk classification"""
    try:
        if 'delivery_risk' not in models:
            return None, None
        
        cols = ['Days for shipping (real)', 'Days for shipment (scheduled)', 'Shipping Mode', 'Order Item Quantity']
        if not all(col in df.columns for col in cols):
            return None, None
        
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
        }, "Delivery risk prediction completed successfully"
    except Exception as e:
        return None, f"Error in delivery risk: {str(e)}"

def run_delay_prediction(df):
    """Run delay regression"""
    try:
        if 'delay_regression' not in models:
            return None, None
        
        cols = ['Days for shipping (real)', 'Days for shipment (scheduled)', 'Shipping Mode', 'Order Item Quantity']
        if not all(col in df.columns for col in cols):
            return None, None
        
        model = models['delay_regression']
        pred = model.predict(df[cols])
        
        return {
            'predictions': pred,
            'avg_delay': np.mean(pred),
            'max_delay': np.max(pred),
            'min_delay': np.min(pred)
        }, "Delay prediction completed successfully"
    except Exception as e:
        return None, f"Error in delay prediction: {str(e)}"

def run_churn_prediction(df):
    """Run customer churn prediction"""
    try:
        if 'churn' not in models:
            return None, None
        
        cols = ['Customer Segment', 'Type', 'Category Name', 'Order Item Quantity', 'Sales', 'Order Profit Per Order']
        if not all(col in df.columns for col in cols):
            return None, None
        
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
        }, "Churn prediction completed successfully"
    except Exception as e:
        return None, f"Error in churn prediction: {str(e)}"

def run_supplier_prediction(df):
    """Run supplier reliability prediction"""
    try:
        if 'supplier' not in models:
            return None, None
        
        cols = ['Days for shipping (real)', 'Days for shipment (scheduled)', 'Shipping Mode', 'Order Item Quantity']
        if not all(col in df.columns for col in cols):
            return None, None
        
        model = models['supplier']
        pred = model.predict(df[cols])
        
        return {
            'predictions': pred,
            'avg_score': np.mean(pred),
            'max_score': np.max(pred),
            'min_score': np.min(pred)
        }, "Supplier reliability prediction completed successfully"
    except Exception as e:
        return None, f"Error in supplier prediction: {str(e)}"

def generate_ai_insight(query, context):
    """Generate intelligent AI response with context"""
    if not gemini_model or not context:
        return "I need more data to provide insights. Please upload relevant data and try again."
    
    try:
        prompt = f"""You are an expert supply chain AI assistant. A user asked: "{query}"

Based on the model analysis below, provide a clear, actionable business insight in 2-3 sentences:

{context}

Focus on:
- Key findings and risks
- Business impact
- Recommended actions
- Confidence in predictions"""
        
        response = gemini_model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.7, top_p=0.9))
        return response.text
    except:
        return context

# ========== SIDEBAR ==========
st.sidebar.markdown("# 🏭 Supply Chain AI Pro")
st.sidebar.markdown("---")

with st.sidebar:
    st.markdown("### System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"🤖 Gemini: {'✅ ON' if gemini_connected else '❌ OFF'}")
    with col2:
        st.markdown(f"📦 Models: {'✅ OK' if models else '❌ ERROR'}")
    
    st.markdown("---")
    page = st.radio("📍 Select Module:", [
        "🏠 Home",
        "📦 Delivery Risk",
        "📈 Demand Forecast",
        "👥 Churn & Supplier",
        "🤖 Smart Chatbot"
    ], label_visibility="collapsed")

# ========== HOME PAGE ==========
if page == "🏠 Home":
    st.markdown('<div class="header-box"><h1>🏭 Supply Chain Risk Intelligence Platform</h1></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("### 🎯 Advanced ML-Powered Analytics")
        st.markdown("""
        **Predict, Analyze & Optimize Your Supply Chain**
        
        Our platform combines cutting-edge machine learning with real-time analytics to help you:
        
        - 🚚 **Delivery Risk Assessment** - Identify at-risk shipments before they fail
        - 📊 **Demand Forecasting** - Accurate predictions with Prophet & LSTM
        - 👥 **Customer Churn** - Retain valuable customers with early warnings
        - ⭐ **Supplier Reliability** - Score and manage supplier performance
        - 🤖 **AI Chatbot** - Natural language insights powered by Gemini 2.5
        """)
    
    with col2:
        st.markdown("### 📊 Dashboard Status")
        st.markdown(f"""
        <div class="metric-card">
        <h3>✅ Models Loaded</h3>
        <p>{len(models)}/4 models active</p>
        </div>
        
        <div class="metric-card">
        <h3>🤖 AI Engine</h3>
        <p>{'Gemini 1.5 Ready' if gemini_connected else 'Limited Mode'}</p>
        </div>
        
        <div class="metric-card">
        <h3>⚡ Performance</h3>
        <p>Real-time Processing</p>
        </div>
        """, unsafe_allow_html=True)

# ========== MODULE 1: DELIVERY RISK ==========
elif page == "📦 Delivery Risk":
    st.markdown('<div class="header-box"><h1>📦 Delivery & Delay Risk Prediction</h1></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("### 📤 Upload Data")
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv', 'xlsx'], key='delivery_upload')
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state['uploaded_data']['delivery'] = df
            st.markdown(f'<div class="success-box">✅ Loaded {len(df)} records</div>', unsafe_allow_html=True)
            st.dataframe(df.head(3), use_container_width=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🚀 Predict Risk", use_container_width=True, key="btn_delivery_risk"):
                    result, msg = run_delivery_risk_prediction(df)
                    if result:
                        st.session_state['predictions']['delivery_risk'] = result
                        st.session_state['model_outputs']['delivery_risk'] = msg
                        st.rerun()
                    else:
                        st.markdown(f'<div class="error-box">❌ {msg}</div>', unsafe_allow_html=True)
            
            with col_b:
                if st.button("📅 Predict Delay", use_container_width=True, key="btn_delay"):
                    result, msg = run_delay_prediction(df)
                    if result:
                        st.session_state['predictions']['delay'] = result
                        st.session_state['model_outputs']['delay'] = msg
                        st.rerun()
                    else:
                        st.markdown(f'<div class="error-box">❌ {msg}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Prediction Results")
        
        if 'delivery_risk' in st.session_state['predictions']:
            result = st.session_state['predictions']['delivery_risk']
            
            st.markdown('<div class="header-box"><h3>🎯 Late Delivery Risk</h3></div>', unsafe_allow_html=True)
            
            col_i, col_ii, col_iii = st.columns(3)
            col_i.metric("🔴 High Risk", result['risk_count'], f"{result['risk_pct']:.1f}%")
            col_ii.metric("🟢 On-Time", result['total'] - result['risk_count'], f"{100-result['risk_pct']:.1f}%")
            col_iii.metric("📦 Total", result['total'], "100%")
            
            # Detailed table
            st.dataframe({
                'Order': range(len(result['predictions'])),
                'Status': ['🔴 HIGH RISK' if p == 1 else '🟢 ON TIME' for p in result['predictions']],
                'Confidence': [f"{max(result['probabilities'][i])*100:.1f}%" for i in range(len(result['predictions']))]
            }, use_container_width=True, hide_index=True)
        
        if 'delay' in st.session_state['predictions']:
            result = st.session_state['predictions']['delay']
            
            st.markdown('<div class="header-box"><h3>⏱️ Delay Forecast</h3></div>', unsafe_allow_html=True)
            
            col_i, col_ii, col_iii = st.columns(3)
            col_i.metric("📊 Avg Delay", f"{result['avg_delay']:.2f}", "days")
            col_ii.metric("⬆️ Max Delay", f"{result['max_delay']:.2f}", "days")
            col_iii.metric("⬇️ Min Delay", f"{result['min_delay']:.2f}", "days")
            
            # Histogram
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(result['predictions'], bins=15, color='#667eea', edgecolor='white', linewidth=1.5)
            ax.set_facecolor('#1e293b')
            fig.patch.set_facecolor('#1e293b')
            ax.set_xlabel('Delay (Days)', color='white')
            ax.set_ylabel('Frequency', color='white')
            ax.set_title('Delivery Delay Distribution', color='white', fontsize=14, fontweight='bold')
            ax.tick_params(colors='white')
            plt.tight_layout()
            st.pyplot(fig)

# ========== MODULE 2: DEMAND FORECAST ==========
elif page == "📈 Demand Forecast":
    st.markdown('<div class="header-box"><h1>📈 Demand & Inventory Forecasting</h1></div>', unsafe_allow_html=True)
    
    st.markdown("### 📤 Upload Time Series Data")
    uploaded_file = st.file_uploader("Choose CSV (date, store, item, sales)", type=['csv', 'xlsx'], key='demand_upload')
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.session_state['uploaded_data']['demand'] = df
        st.markdown(f'<div class="success-box">✅ Loaded {len(df)} records</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔮 Prophet Forecast (7 days)", use_container_width=True):
                try:
                    if 'date' in df.columns and 'sales' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        day_sales = df.groupby('date')['sales'].sum().reset_index()
                        prophet_df = day_sales.rename(columns={"date": "ds", "sales": "y"})
                        
                        with st.spinner("🔄 Training Prophet..."):
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
            recent_forecast = forecast.tail(7)
            
            st.markdown('<div class="header-box"><h3>📊 7-Day Demand Forecast</h3></div>', unsafe_allow_html=True)
            
            col_i, col_ii, col_iii = st.columns(3)
            col_i.metric("📈 Avg Forecast", f"{recent_forecast['yhat'].mean():.0f}", "units")
            col_ii.metric("⬆️ Peak", f"{recent_forecast['yhat'].max():.0f}", "units")
            col_iii.metric("📊 Range", f"±{recent_forecast['yhat_upper'].mean() - recent_forecast['yhat'].mean():.0f}", "units")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='#667eea', linewidth=2.5)
            ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2, color='#667eea')
            ax.set_facecolor('#1e293b')
            fig.patch.set_facecolor('#1e293b')
            ax.set_xlabel('Date', color='white')
            ax.set_ylabel('Demand (Units)', color='white')
            ax.set_title('Demand Forecast with 95% Confidence Interval', color='white', fontsize=14, fontweight='bold')
            ax.legend(loc='upper left', facecolor='#1e293b', edgecolor='white')
            ax.tick_params(colors='white')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

# ========== MODULE 3: CHURN & SUPPLIER ==========
elif page == "👥 Churn & Supplier":
    st.markdown('<div class="header-box"><h1>👥 Customer Churn & Supplier Reliability</h1></div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔴 Customer Churn", "⭐ Supplier Reliability"])
    
    with tab1:
        st.markdown("### 📤 Upload Customer Data")
        uploaded_file = st.file_uploader("Choose CSV", type=['csv', 'xlsx'], key='churn_upload')
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state['uploaded_data']['churn'] = df
            st.markdown(f'<div class="success-box">✅ Loaded {len(df)} records</div>', unsafe_allow_html=True)
            
            if st.button("🎯 Predict Customer Churn", use_container_width=True):
                result, msg = run_churn_prediction(df)
                if result:
                    st.session_state['predictions']['churn'] = result
                    st.rerun()
                else:
                    st.markdown(f'<div class="error-box">❌ {msg}</div>', unsafe_allow_html=True)
            
            if 'churn' in st.session_state['predictions']:
                result = st.session_state['predictions']['churn']
                
                st.markdown('<div class="header-box"><h3>📊 Churn Analysis</h3></div>', unsafe_allow_html=True)
                
                col_i, col_ii, col_iii = st.columns(3)
                col_i.metric("🔴 At-Risk", result['churn_count'], f"{result['churn_pct']:.1f}%")
                col_ii.metric("🟢 Retained", result['total'] - result['churn_count'], f"{100-result['churn_pct']:.1f}%")
                col_iii.metric("👥 Total", result['total'], "100%")
    
    with tab2:
        st.markdown("### 📤 Upload Supplier Data")
        uploaded_file = st.file_uploader("Choose CSV", type=['csv', 'xlsx'], key='supplier_upload')
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state['uploaded_data']['supplier'] = df
            st.markdown(f'<div class="success-box">✅ Loaded {len(df)} records</div>', unsafe_allow_html=True)
            
            if st.button("⭐ Predict Supplier Reliability", use_container_width=True):
                result, msg = run_supplier_prediction(df)
                if result:
                    st.session_state['predictions']['supplier'] = result
                    st.rerun()
                else:
                    st.markdown(f'<div class="error-box">❌ {msg}</div>', unsafe_allow_html=True)
            
            if 'supplier' in st.session_state['predictions']:
                result = st.session_state['predictions']['supplier']
                
                st.markdown('<div class="header-box"><h3>⭐ Reliability Scores</h3></div>', unsafe_allow_html=True)
                
                col_i, col_ii, col_iii = st.columns(3)
                col_i.metric("📊 Avg Score", f"{result['avg_score']:.2f}", "/10")
                col_ii.metric("⬆️ Best", f"{result['max_score']:.2f}", "/10")
                col_iii.metric("⬇️ Worst", f"{result['min_score']:.2f}", "/10")

# ========== SMART CHATBOT PAGE ==========
elif page == "🤖 Smart Chatbot":
    st.markdown('<div class="header-box"><h1>🤖 AI Supply Chain Chatbot</h1></div>', unsafe_allow_html=True)
    
    st.info("💡 **Upload data in other modules first. The chatbot will analyze all your predictions and answer any supply chain questions!**")
    
    # Data upload for chatbot
    st.markdown("### 📂 Upload Data for Chatbot Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        uploaded_file = st.file_uploader("Delivery Data", type=['csv', 'xlsx'], key='chat_delivery')
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state['uploaded_data']['chat_delivery'] = df
            st.markdown('<div class="success-box">✅ Delivery data ready</div>', unsafe_allow_html=True)
    
    with col2:
        uploaded_file = st.file_uploader("Demand Data", type=['csv', 'xlsx'], key='chat_demand')
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state['uploaded_data']['chat_demand'] = df
            st.markdown('<div class="success-box">✅ Demand data ready</div>', unsafe_allow_html=True)
    
    with col3:
        uploaded_file = st.file_uploader("Customer Data", type=['csv', 'xlsx'], key='chat_customer')
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state['uploaded_data']['chat_customer'] = df
            st.markdown('<div class="success-box">✅ Customer data ready</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💬 Ask Your Supply Chain Questions")
    
    user_input = st.text_area("Your Question:", placeholder="E.g., 'What is the average delivery delay?', 'How many customers are at risk of churning?'", height=100)
    
    if st.button("🔍 Analyze & Answer", use_container_width=True):
        if user_input:
            with st.spinner("🤖 Analyzing..."):
                # Multi-module analysis
                analysis_context = []
                
                # Check delivery data
                if 'chat_delivery' in st.session_state['uploaded_data']:
                    df = st.session_state['uploaded_data']['chat_delivery']
                    result, _ = run_delivery_risk_prediction(df)
                    if result:
                        analysis_context.append(f"Delivery Analysis: {result['risk_count']} high-risk orders out of {result['total']} ({result['risk_pct']:.1f}%)")
                    
                    result, _ = run_delay_prediction(df)
                    if result:
                        analysis_context.append(f"Average delay: {result['avg_delay']:.2f} days (Range: {result['min_delay']:.2f}-{result['max_delay']:.2f} days)")
                
                # Check demand data
                if 'chat_demand' in st.session_state['uploaded_data']:
                    df = st.session_state['uploaded_data']['chat_demand']
                    if 'date' in df.columns and 'sales' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        day_sales = df.groupby('date')['sales'].sum().reset_index()
                        prophet_df = day_sales.rename(columns={"date": "ds", "sales": "y"})
                        model = Prophet(interval_width=0.95)
                        model.fit(prophet_df)
                        future = model.make_future_dataframe(periods=7)
                        forecast = model.predict(future)
                        analysis_context.append(f"7-day demand forecast: Average {forecast.tail(7)['yhat'].mean():.0f} units")
                
                # Check customer data
                if 'chat_customer' in st.session_state['uploaded_data']:
                    df = st.session_state['uploaded_data']['chat_customer']
                    result, _ = run_churn_prediction(df)
                    if result:
                        analysis_context.append(f"Customer churn: {result['churn_count']} at-risk customers ({result['churn_pct']:.1f}%)")
                
                context = " | ".join(analysis_context) if analysis_context else "No data analyzed"
                
                # Generate AI response
                ai_response = generate_ai_insight(user_input, context)
                
                # Add to chat history
                st.session_state['chat_history'].append({"role": "user", "content": user_input})
                st.session_state['chat_history'].append({"role": "ai", "content": ai_response})
    
    # Display chat history
    st.markdown("---")
    st.markdown("### 📝 Conversation")
    
    for entry in st.session_state['chat_history'][-6:]:  # Show last 6 messages
        if entry["role"] == "user":
            st.markdown(f'<div class="chat-message-user">👤 <b>You:</b> {entry["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message-ai">🤖 <b>AI:</b> {entry["content"]}</div>', unsafe_allow_html=True)

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #667eea; margin-top: 50px;'>
    <p>🏭 Supply Chain AI Pro v2.0 | Powered by CatBoost, Prophet, LSTM & Gemini</p>
    <p>© 2025 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
