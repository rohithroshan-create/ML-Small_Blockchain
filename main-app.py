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
    initial_sidebar_state="collapsed"
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
        background: white; border: 2px solid #ff6b6b; padding: 20px;
        border-radius: 12px; margin: 10px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
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
    st.session_state['chat_history'] = {'delivery': [], 'demand': [], 'churn': []}
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
            return None, f"Missing required columns"
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
            return None, f"Missing required columns"
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
            return None, f"Missing required columns"
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
            return None, f"Missing required columns"
        model = models['supplier']
        pred = model.predict(df[cols])
        return {'predictions': pred, 'avg_score': np.mean(pred), 'max_score': np.max(pred), 'min_score': np.min(pred)}, "Success"
    except Exception as e:
        return None, str(e)

def get_intelligent_response(user_question, module_name, predictions_context=""):
    """Get intelligent AI response that analyzes predictions"""
    if not gemini_model:
        return None
    
    try:
        prompt = f"""You are an expert supply chain consultant analyzing {module_name} data.

User Question: {user_question}

Current Predictions/Context: {predictions_context}

IMPORTANT INSTRUCTIONS:
1. If user asks about predictions, ANALYZE the data provided and explain:
   - What the predictions mean
   - Why these results occurred
   - Actionable steps to improve

2. If user asks "how to reduce/improve/fix X":
   - Reference the prediction data provided
   - Give specific, data-driven recommendations
   - Explain expected outcomes with timelines

3. If user asks general questions:
   - Answer from your supply chain expertise
   - Relate back to the module if possible
   - Be practical and actionable

4. Always be conversational, NOT robotic
5. If predictions show problems, suggest solutions
6. If no context provided, answer generally

Answer the user's question directly and helpfully."""
        
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.9, top_p=0.95, max_output_tokens=1500)
        )
        return response.text
    except:
        return None

# ========== HEADER ==========
st.markdown('<div class="header-box"><h1>🏭 Supply Chain AI Pro</h1><p>Advanced ML Predictions with Intelligent AI Chatbot</p></div>', unsafe_allow_html=True)

# ========== TABS ==========
tab1, tab2, tab3 = st.tabs(["📦 Delivery Module", "📈 Demand Module", "👥 Churn & Supplier Module"])

# ========== TAB 1: DELIVERY MODULE ==========
with tab1:
    st.markdown("### 📦 Late Delivery Risk & Delay Prediction")
    
    col_upload, col_results = st.columns([1, 1.3])
    
    with col_upload:
        st.markdown("**Upload & Predict**")
        uploaded_file = st.file_uploader("Upload Delivery Data (CSV/XLSX)", type=['csv', 'xlsx'], key='delivery_file')
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state['uploaded_data']['delivery'] = df
            st.markdown(f'<div class="success-box">✅ Loaded {len(df)} records</div>', unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔴 Risk Prediction", use_container_width=True):
                    result, msg = run_delivery_risk_prediction(df)
                    if result:
                        st.session_state['predictions']['delivery_risk'] = result
                        st.rerun()
                    else:
                        st.markdown(f'<div class="error-box">❌ {msg}</div>', unsafe_allow_html=True)
            
            with col_b:
                if st.button("⏱️ Delay Prediction", use_container_width=True):
                    result, msg = run_delay_prediction(df)
                    if result:
                        st.session_state['predictions']['delay'] = result
                        st.rerun()
                    else:
                        st.markdown(f'<div class="error-box">❌ {msg}</div>', unsafe_allow_html=True)
    
    with col_results:
        st.markdown("**Prediction Results**")
        
        if 'delivery_risk' in st.session_state['predictions']:
            result = st.session_state['predictions']['delivery_risk']
            col_i, col_ii, col_iii = st.columns(3)
            col_i.metric("🔴 High Risk", result['risk_count'], f"{result['risk_pct']:.1f}%")
            col_ii.metric("🟢 On-Time", result['total'] - result['risk_count'], f"{100-result['risk_pct']:.1f}%")
            col_iii.metric("📦 Total", result['total'], "100%")
            
            st.dataframe({
                'Order': range(min(5, len(result['predictions']))),
                'Status': ['🔴 HIGH RISK' if result['predictions'][i] == 1 else '🟢 ON TIME' for i in range(min(5, len(result['predictions'])))],
                'Confidence': [f"{max(result['probabilities'][i])*100:.1f}%" for i in range(min(5, len(result['predictions'])))]
            }, use_container_width=True, hide_index=True)
        
        if 'delay' in st.session_state['predictions']:
            result = st.session_state['predictions']['delay']
            col_i, col_ii, col_iii = st.columns(3)
            col_i.metric("📊 Avg", f"{result['avg_delay']:.2f}", "days")
            col_ii.metric("⬆️ Max", f"{result['max_delay']:.2f}", "days")
            col_iii.metric("⬇️ Min", f"{result['min_delay']:.2f}", "days")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(result['predictions'], bins=15, color='#ff6b6b', edgecolor='white', linewidth=1.5)
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_xlabel('Delay (Days)', color='#1a1a1a')
            ax.set_ylabel('Frequency', color='#1a1a1a')
            ax.set_title('Delivery Delay Distribution', color='#1a1a1a')
            ax.tick_params(colors='#1a1a1a')
            plt.tight_layout()
            st.pyplot(fig)
    
    # CHATBOT SECTION
    st.markdown("---")
    st.markdown("### 💬 AI Assistant - Ask Anything About Delivery")
    
    # Build prediction context
    pred_context = ""
    if 'delivery_risk' in st.session_state['predictions']:
        r = st.session_state['predictions']['delivery_risk']
        pred_context += f"Delivery Risk: {r['risk_count']}/{r['total']} high-risk ({r['risk_pct']:.1f}%). "
    if 'delay' in st.session_state['predictions']:
        r = st.session_state['predictions']['delay']
        pred_context += f"Average delay: {r['avg_delay']:.2f} days (Range: {r['min_delay']:.2f}-{r['max_delay']:.2f} days)."
    
    user_input = st.text_input("Ask your question:", placeholder="E.g., 'Why is delivery delayed?', 'How to reduce delays?', 'What's causing high risk?'", key="delivery_chat")
    
    if user_input:
        with st.spinner("🤖 Analyzing..."):
            response = get_intelligent_response(user_input, "Delivery Module", pred_context)
            
            if not response:
                response = f"I'm analyzing your question about delivery. Context: {pred_context if pred_context else 'No predictions yet. Upload data and make predictions first for detailed analysis.'}"
            
            st.session_state['chat_history']['delivery'].append({"role": "user", "content": user_input})
            st.session_state['chat_history']['delivery'].append({"role": "ai", "content": response})
    
    # Display chat history
    for msg in st.session_state['chat_history']['delivery']:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message-user">👤 <b>You:</b><br/>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message-ai">🤖 <b>AI:</b><br/>{msg["content"]}</div>', unsafe_allow_html=True)

# ========== TAB 2: DEMAND MODULE ==========
with tab2:
    st.markdown("### 📈 Demand Forecasting (Prophet)")
    
    col_upload, col_results = st.columns([1, 1.3])
    
    with col_upload:
        st.markdown("**Upload & Predict**")
        uploaded_file = st.file_uploader("Upload Demand Data (date, sales)", type=['csv', 'xlsx'], key='demand_file')
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state['uploaded_data']['demand'] = df
            st.markdown(f'<div class="success-box">✅ Loaded {len(df)} records</div>', unsafe_allow_html=True)
            
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
                    st.markdown(f'<div class="error-box">❌ {str(e)}</div>', unsafe_allow_html=True)
    
    with col_results:
        st.markdown("**Forecast Results**")
        
        if 'prophet' in st.session_state['predictions']:
            forecast = st.session_state['predictions']['prophet']
            recent = forecast.tail(7)
            
            col_i, col_ii, col_iii = st.columns(3)
            col_i.metric("📈 Avg", f"{recent['yhat'].mean():.0f}", "units")
            col_ii.metric("⬆️ Peak", f"{recent['yhat'].max():.0f}", "units")
            col_iii.metric("Range", f"±{recent['yhat_upper'].mean() - recent['yhat'].mean():.0f}", "units")
            
            fig, ax = plt.subplots(figsize=(10, 4))
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
    
    # CHATBOT SECTION
    st.markdown("---")
    st.markdown("### 💬 AI Assistant - Ask About Demand")
    
    pred_context = ""
    if 'prophet' in st.session_state['predictions']:
        f = st.session_state['predictions']['prophet']
        recent = f.tail(7)
        pred_context = f"Forecast avg: {recent['yhat'].mean():.0f} units, Peak: {recent['yhat'].max():.0f}, Range: ±{recent['yhat_upper'].mean() - recent['yhat'].mean():.0f}"
    
    user_input = st.text_input("Ask your question:", placeholder="E.g., 'What's the demand trend?', 'How to prepare inventory?'", key="demand_chat")
    
    if user_input:
        with st.spinner("🤖 Analyzing..."):
            response = get_intelligent_response(user_input, "Demand Module", pred_context)
            
            if not response:
                response = f"Analyzing demand question. Context: {pred_context if pred_context else 'Upload and forecast data first.'}"
            
            st.session_state['chat_history']['demand'].append({"role": "user", "content": user_input})
            st.session_state['chat_history']['demand'].append({"role": "ai", "content": response})
    
    for msg in st.session_state['chat_history']['demand']:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message-user">👤 <b>You:</b><br/>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message-ai">🤖 <b>AI:</b><br/>{msg["content"]}</div>', unsafe_allow_html=True)

# ========== TAB 3: CHURN & SUPPLIER MODULE ==========
with tab3:
    st.markdown("### 👥 Customer Churn & ⭐ Supplier Reliability")
    
    churn_col, supplier_col = st.columns(2)
    
    # CHURN SECTION
    with churn_col:
        st.markdown("**Churn Prediction**")
        uploaded_file = st.file_uploader("Upload Customer Data", type=['csv', 'xlsx'], key='churn_file')
        
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
                col_i, col_ii = st.columns(2)
                col_i.metric("🔴 At-Risk", result['churn_count'], f"{result['churn_pct']:.1f}%")
                col_ii.metric("🟢 Retained", result['total'] - result['churn_count'], f"{100-result['churn_pct']:.1f}%")
    
    # SUPPLIER SECTION
    with supplier_col:
        st.markdown("**Supplier Reliability**")
        uploaded_file = st.file_uploader("Upload Supplier Data", type=['csv', 'xlsx'], key='supplier_file')
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state['uploaded_data']['supplier'] = df
            st.markdown(f'<div class="success-box">✅ Loaded {len(df)} records</div>', unsafe_allow_html=True)
            
            if st.button("⭐ Score Suppliers", use_container_width=True):
                result, msg = run_supplier_prediction(df)
                if result:
                    st.session_state['predictions']['supplier'] = result
                    st.rerun()
                else:
                    st.markdown(f'<div class="error-box">❌ {msg}</div>', unsafe_allow_html=True)
            
            if 'supplier' in st.session_state['predictions']:
                result = st.session_state['predictions']['supplier']
                col_i, col_ii = st.columns(2)
                col_i.metric("📊 Avg Score", f"{result['avg_score']:.2f}", "/10")
                col_ii.metric("⭐ Best", f"{result['max_score']:.2f}", "/10")
    
    # CHATBOT SECTION
    st.markdown("---")
    st.markdown("### 💬 AI Assistant - Churn & Supplier Questions")
    
    pred_context = ""
    if 'churn' in st.session_state['predictions']:
        c = st.session_state['predictions']['churn']
        pred_context += f"Churn Risk: {c['churn_count']}/{c['total']} ({c['churn_pct']:.1f}%). "
    if 'supplier' in st.session_state['predictions']:
        s = st.session_state['predictions']['supplier']
        pred_context += f"Supplier Reliability: Avg {s['avg_score']:.2f}/10."
    
    user_input = st.text_input("Ask your question:", placeholder="E.g., 'How to reduce churn?', 'Which suppliers are reliable?'", key="churn_chat")
    
    if user_input:
        with st.spinner("🤖 Analyzing..."):
            response = get_intelligent_response(user_input, "Churn & Supplier Module", pred_context)
            
            if not response:
                response = f"Analyzing your query. Context: {pred_context if pred_context else 'Upload and predict data first.'}"
            
            st.session_state['chat_history']['churn'].append({"role": "user", "content": user_input})
            st.session_state['chat_history']['churn'].append({"role": "ai", "content": response})
    
    for msg in st.session_state['chat_history']['churn']:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message-user">👤 <b>You:</b><br/>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message-ai">🤖 <b>AI:</b><br/>{msg["content"]}</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #ff6b6b;'><p>🏭 Supply Chain AI Pro v6.0 | 3 Modules + Intelligent Chatbot | Production Ready</p></div>", unsafe_allow_html=True)
