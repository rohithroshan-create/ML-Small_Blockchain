import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="🏭 Supply Chain AI", page_icon="🏭", layout="wide", initial_sidebar_state="collapsed")

# CSS STYLING
st.markdown("""
    <style>
    * { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    body { background: #f8f9fa; }
    .stApp { background: #f8f9fa; }
    .header-box { 
        background: linear-gradient(135deg, #ff6b6b 0%, #c44569 100%); 
        color: white; padding: 30px; border-radius: 15px; margin: 20px 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .header-box h1 { color: white; margin: 0; font-size: 2.5em; }
    .success-box { background: #4caf50; color: white; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid white; font-weight: bold; }
    .error-box { background: #ff6b6b; color: white; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid white; font-weight: bold; }
    .info-box { background: #2196F3; color: white; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid white; }
    .chat-user { background: #2196F3; color: white; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid white; }
    .chat-bot { background: #4caf50; color: white; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid white; }
    .chat-reject { background: #ff9800; color: white; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid white; }
    .metric-box { background: white; border: 2px solid #ff6b6b; padding: 20px; border-radius: 12px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# SESSION STATE
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = {}
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# LOAD MODELS - RANDOM FOREST PIPELINES
@st.cache_resource
def load_models():
    models = {}
    try:
        # Random Forest Classifier Pipeline (for classification tasks)
        if os.path.exists("rf_classifier_pipeline.pkl"):
            models['rf_classifier'] = joblib.load("rf_classifier_pipeline.pkl")
            st.sidebar.success("✅ RF Classifier loaded")
        
        # Random Forest Regressor Pipeline (for regression tasks)
        if os.path.exists("rf_regressor_pipeline.pkl"):
            models['rf_regressor'] = joblib.load("rf_regressor_pipeline.pkl")
            st.sidebar.success("✅ RF Regressor loaded")
        
        # Optional CatBoost models if available
        if os.path.exists("catboost_delay_regression.pkl"):
            models['delay_regression'] = joblib.load("catboost_delay_regression.pkl")
        if os.path.exists("catboost_customer_churn.pkl"):
            models['churn'] = joblib.load("catboost_customer_churn.pkl")
        if os.path.exists("catboost_supplier_reliability.pkl"):
            models['supplier'] = joblib.load("catboost_supplier_reliability.pkl")
    except Exception as e:
        st.sidebar.error(f"Error loading models: {str(e)}")
    
    if not models:
        st.sidebar.warning("⚠️ No models found. Ensure .pkl files are in project root.")
    
    return models

models = load_models()

# PREDICTION FUNCTIONS
def run_delivery_prediction(df):
    """Run delivery prediction using Random Forest Classifier"""
    try:
        if 'rf_classifier' not in models:
            return None, "RF Classifier model not found"
        
        # Try to predict with the pipeline
        pred = models['rf_classifier'].predict(df)
        proba = models['rf_classifier'].predict_proba(df) if hasattr(models['rf_classifier'], 'predict_proba') else None
        
        if pred is not None:
            risk_count = sum(pred)
            total = len(pred)
            risk_pct = (risk_count / total * 100) if total > 0 else 0
            return {
                'pred': pred, 
                'proba': proba,
                'risk': risk_count, 
                'total': total, 
                'pct': risk_pct
            }, "Success"
        else:
            return None, "Prediction failed"
    except Exception as e:
        return None, f"Error: {str(e)}"

def run_delay_prediction(df):
    """Run delay prediction using Random Forest Regressor"""
    try:
        if 'rf_regressor' not in models:
            return None, "RF Regressor model not found"
        
        pred = models['rf_regressor'].predict(df)
        
        if pred is not None:
            return {
                'pred': pred,
                'avg': np.mean(pred),
                'max': np.max(pred),
                'min': np.min(pred)
            }, "Success"
        else:
            return None, "Prediction failed"
    except Exception as e:
        return None, f"Error: {str(e)}"

def run_churn_prediction(df):
    """Run churn prediction"""
    try:
        if 'churn' not in models:
            return None, "Churn model not found"
        
        pred = models['churn'].predict(df)
        proba = models['churn'].predict_proba(df) if hasattr(models['churn'], 'predict_proba') else None
        
        churn_count = sum(pred)
        total = len(pred)
        churn_pct = (churn_count / total * 100) if total > 0 else 0
        
        return {
            'pred': pred,
            'proba': proba,
            'churn': churn_count,
            'total': total,
            'pct': churn_pct
        }, "Success"
    except Exception as e:
        return None, f"Error: {str(e)}"

def run_supplier_prediction(df):
    """Run supplier reliability prediction"""
    try:
        if 'supplier' not in models:
            return None, "Supplier model not found"
        
        pred = models['supplier'].predict(df)
        
        if pred is not None:
            return {
                'pred': pred,
                'avg': np.mean(pred),
                'max': np.max(pred),
                'min': np.min(pred)
            }, "Success"
        else:
            return None, "Prediction failed"
    except Exception as e:
        return None, f"Error: {str(e)}"

# SUPPLY CHAIN CHATBOT FUNCTIONS
def check_supply_chain_relevance(question):
    """Check if question is supply chain/delivery related"""
    keywords = [
        'delivery', 'delay', 'supply', 'chain', 'shipping', 'shipment', 'order', 
        'customer', 'churn', 'supplier', 'demand', 'forecast', 'logistics', 
        'warehouse', 'inventory', 'transport', 'carrier', 'packaging', 'route', 
        'cost', 'time', 'performance', 'risk', 'optimization', 'planning',
        'tips', 'how to', 'improve', 'reduce', 'increase', 'best practice', 
        'management', 'distribution', 'fulfillment', 'procurement'
    ]
    return any(kw in question.lower() for kw in keywords)

def get_supply_chain_response(question):
    """Provide supply chain domain-specific responses"""
    q = question.lower()
    
    if 'delivery' in q and ('how' in q or 'tip' in q or 'improve' in q):
        return """🚚 **Supply Chain Delivery Tips:**

1. **Route Optimization**
   - Use GPS-based routing for fastest paths
   - Consolidate orders by delivery zones
   - Reduce per-order delivery cost by 20-30%

2. **Carrier Selection**
   - Express: 2-3 days (urgent orders)
   - Standard: 5-7 days (regular orders)
   - Compare carrier rates quarterly

3. **Packaging Best Practices**
   - Lightweight packaging reduces costs
   - Proper packaging prevents damage
   - Eco-friendly options improve brand image

4. **Timing Optimization**
   - Ship early in the week for better delivery rates
   - Avoid peak seasons when possible
   - Plan ahead for holiday delays"""
    
    elif 'reduce delay' in q or 'faster delivery' in q or 'speed up' in q:
        return """⚡ **Ways to Reduce Delivery Delays:**

1. **Process Improvements**
   - Streamline order picking (batch by area)
   - Pre-pack common orders
   - Automate labeling and sorting

2. **Supplier Coordination**
   - Set clear SLAs with suppliers
   - Daily communication with logistics
   - Monitor supplier performance metrics

3. **Technology Solutions**
   - Real-time tracking system
   - Automated alerts for delays
   - Predictive analytics for risk orders

4. **Network Optimization**
   - Regional distribution centers
   - Local warehouses for fast-moving items
   - Partner with multiple carriers"""
    
    elif 'supply chain' in q and ('what' in q or 'explain' in q or 'overview' in q):
        return """🏭 **Supply Chain Overview:**

**Key Components:**
1. **Procurement** - Sourcing materials from suppliers
2. **Production** - Manufacturing/processing products
3. **Distribution** - Moving goods to regional hubs
4. **Logistics** - Shipping to customers
5. **Returns** - Processing returns and refunds

**Key Metrics:**
- Lead time (supplier to warehouse)
- Fulfillment time (order to shipment)
- Delivery time (shipment to customer)
- Cost per unit
- Customer satisfaction %

**Goal:** Efficient flow of goods with minimal cost and delay"""
    
    elif 'cost' in q and ('reduce' in q or 'lower' in q or 'save' in q):
        return """💰 **Reducing Supply Chain Costs:**

1. **Shipping Optimization**
   - Negotiate bulk rates (5-15% savings)
   - Use slower shipping when acceptable
   - Consolidate shipments

2. **Inventory Management**
   - Reduce excess stock (lower storage cost)
   - Improve demand forecasting accuracy
   - Use just-in-time delivery methods

3. **Supplier Management**
   - Develop long-term relationships for better pricing
   - Consolidate suppliers (reduce variety)
   - Source from regional suppliers

4. **Process Efficiency**
   - Automate order picking and sorting
   - Reduce manual handling and errors
   - Improve warehouse layout"""
    
    elif 'demand' in q and ('forecast' in q or 'predict' in q or 'planning' in q):
        return """📊 **Demand Forecasting Guide:**

**Purpose:** Predict customer demand to optimize inventory levels

**Forecasting Methods:**
1. Historical data analysis
2. Seasonal adjustments
3. Trend analysis
4. Time-series forecasting (Prophet)

**Benefits:**
- Reduce excess inventory
- Prevent stockouts
- Optimize warehouse space
- Better supplier planning
- Improved cash flow

**Use this app's Demand Module to forecast!**"""
    
    elif 'customer' in q and ('retention' in q or 'loyalty' in q or 'keep' in q):
        return """👥 **Customer Retention Strategy:**

1. **Communication**
   - Track order status in real-time
   - Proactive delay notifications
   - Dedicated customer support

2. **Quality & Service**
   - Consistent on-time delivery
   - Proper packaging prevents damage
   - Easy, hassle-free returns process

3. **Incentives**
   - Loyalty programs for repeat customers
   - Bulk order discounts
   - Free shipping thresholds

4. **Feedback Loop**
   - Survey customers regularly
   - Act on feedback quickly
   - Show continuous improvement"""
    
    elif 'supplier' in q and ('choose' in q or 'select' in q or 'evaluate' in q):
        return """⭐ **Choosing & Evaluating Suppliers:**

**Key Evaluation Criteria:**
1. Reliability (on-time delivery %)
2. Quality (defect rate, standards)
3. Cost competitiveness
4. Communication responsiveness
5. Scalability for growth
6. Location/proximity to operations
7. Financial stability

**Evaluation Process:**
- Request quotes from multiple suppliers
- Check references and past performance
- Review quality certifications
- Trial orders first
- Establish SLAs
- Regular performance reviews"""
    
    elif 'warehouse' in q or 'storage' in q or 'inventory' in q:
        return """📦 **Warehouse & Inventory Management:**

**Inventory Optimization:**
1. ABC analysis (classify items by value)
2. Keep fast-movers readily accessible
3. Implement FIFO (first in, first out)
4. Regular stock audits

**Warehouse Organization:**
- Organize by product type/category
- Clear labeling system
- Efficient picking routes
- Safety protocols and training

**Technology Implementation:**
- Inventory management system
- Barcode/RFID tracking
- Real-time stock visibility
- Automated low-stock alerts"""
    
    elif 'return' in q or 'reverse' in q:
        return """↩️ **Returns & Reverse Logistics:**

**Return Process:**
1. Accept returns within reasonable period
2. Inspect item condition
3. Restock, repair, or dispose appropriately
4. Process refund quickly

**Optimization Tips:**
- Minimize returns through quality control
- Establish return hubs for efficiency
- Partner with carriers for pickups
- Track return metrics and patterns

**Best Practices:**
- Easy, customer-friendly process
- Fast refunds (within 5-7 days)
- Environmental considerations
- Data analysis for improvement"""
    
    else:
        return """💡 **Supply Chain Topics I Can Help With:**

✅ Delivery optimization
✅ Reducing costs
✅ Supply chain overview
✅ Demand forecasting
✅ Warehouse management
✅ Supplier selection
✅ Customer retention
✅ Reverse logistics
✅ Inventory management

Ask me about any supply chain or delivery-related topic!"""

# HEADER
st.markdown('<div class="header-box"><h1>🏭 Supply Chain AI System</h1><p style="margin: 10px 0 0 0; font-size: 1.1em;">ML-Powered Predictions & Delivery Insights</p></div>', unsafe_allow_html=True)

# MAIN TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📦 Delivery", "📈 Demand", "👥 Churn & Supplier", "💬 Chatbot", "ℹ️ Info"])

# ===== TAB 1: DELIVERY PREDICTION =====
with tab1:
    st.markdown("### 📦 Delivery Prediction (Random Forest)")
    
    col_upload, col_results = st.columns([1, 1.3])
    
    with col_upload:
        st.markdown("**Upload Data & Predict**")
        uploaded_file = st.file_uploader("Upload CSV/XLSX", type=['csv', 'xlsx'], key='delivery_upload')
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.markdown(f'<div class="success-box">✅ {len(df)} records loaded | {len(df.columns)} features</div>', unsafe_allow_html=True)
                
                with st.expander("📋 View Data Sample"):
                    st.dataframe(df.head(3), use_container_width=True)
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    if st.button("🔴 Risk Prediction", use_container_width=True, key="delivery_risk_btn"):
                        result, msg = run_delivery_prediction(df)
                        if result:
                            st.session_state['predictions']['delivery_risk'] = result
                            st.rerun()
                        else:
                            st.markdown(f'<div class="error-box">❌ {msg}</div>', unsafe_allow_html=True)
                
                with col_b:
                    if st.button("⏱️ Delay Prediction", use_container_width=True, key="delay_btn"):
                        result, msg = run_delay_prediction(df)
                        if result:
                            st.session_state['predictions']['delay'] = result
                            st.rerun()
                        else:
                            st.markdown(f'<div class="error-box">❌ {msg}</div>', unsafe_allow_html=True)
            
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Error loading file: {str(e)}</div>', unsafe_allow_html=True)
    
    with col_results:
        st.markdown("**Prediction Results**")
        
        if 'delivery_risk' in st.session_state['predictions']:
            r = st.session_state['predictions']['delivery_risk']
            
            col_i, col_ii, col_iii = st.columns(3)
            with col_i:
                st.metric("🔴 High Risk", r['risk'], f"{r['pct']:.1f}%")
            with col_ii:
                st.metric("🟢 On-Time", r['total']-r['risk'], f"{100-r['pct']:.1f}%")
            with col_iii:
                st.metric("📦 Total", r['total'], "100%")
            
            if r['proba'] is not None and len(r['pred']) > 0:
                result_df = pd.DataFrame({
                    'Order_ID': range(min(5, len(r['pred']))),
                    'Status': ['🔴 HIGH RISK' if r['pred'][i] == 1 else '🟢 ON-TIME' for i in range(min(5, len(r['pred'])))],
                    'Confidence': [f"{max(r['proba'][i])*100:.1f}%" for i in range(min(5, len(r['proba'])))]
                })
                st.dataframe(result_df, use_container_width=True, hide_index=True)
        
        if 'delay' in st.session_state['predictions']:
            d = st.session_state['predictions']['delay']
            
            col_i, col_ii, col_iii = st.columns(3)
            with col_i:
                st.metric("📊 Avg Delay", f"{d['avg']:.2f}", "days")
            with col_ii:
                st.metric("⬆️ Max", f"{d['max']:.2f}", "days")
            with col_iii:
                st.metric("⬇️ Min", f"{d['min']:.2f}", "days")
            
            # Histogram
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(d['pred'], bins=15, color='#ff6b6b', edgecolor='white', linewidth=1.5, alpha=0.8)
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_xlabel('Delay (Days)', fontsize=11, color='#1a1a1a')
            ax.set_ylabel('Frequency', fontsize=11, color='#1a1a1a')
            ax.set_title('Delivery Delay Distribution', fontsize=12, fontweight='bold', color='#1a1a1a')
            ax.tick_params(colors='#1a1a1a')
            plt.tight_layout()
            st.pyplot(fig)

# ===== TAB 2: DEMAND FORECASTING =====
with tab2:
    st.markdown("### 📈 Demand Forecasting (Prophet)")
    
    col_upload, col_results = st.columns([1, 1.3])
    
    with col_upload:
        st.markdown("**Upload Time Series Data**")
        st.markdown("*Required columns: date, sales*")
        uploaded_file = st.file_uploader("Upload CSV/XLSX", type=['csv', 'xlsx'], key='demand_upload')
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.markdown(f'<div class="success-box">✅ {len(df)} records loaded</div>', unsafe_allow_html=True)
                
                if st.button("🔮 Forecast 7 Days", use_container_width=True):
                    with st.spinner("⏳ Training Prophet model..."):
                        try:
                            df['date'] = pd.to_datetime(df['date'])
                            day_sales = df.groupby('date')['sales'].sum().reset_index().sort_values('date')
                            pdf = day_sales.rename(columns={"date": "ds", "sales": "y"})
                            
                            model = Prophet(
                                yearly_seasonality=True,
                                weekly_seasonality=True,
                                daily_seasonality=False,
                                interval_width=0.95,
                                changepoint_prior_scale=0.05
                            )
                            model.fit(pdf)
                            future = model.make_future_dataframe(periods=7)
                            forecast = model.predict(future)
                            
                            st.session_state['predictions']['prophet'] = forecast
                            st.success("✅ Forecast complete!")
                            st.rerun()
                        
                        except Exception as e:
                            st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
            
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Error loading file: {str(e)}</div>', unsafe_allow_html=True)
    
    with col_results:
        st.markdown("**Forecast Results**")
        
        if 'prophet' in st.session_state['predictions']:
            f = st.session_state['predictions']['prophet']
            recent = f.tail(7)
            
            col_i, col_ii, col_iii = st.columns(3)
            with col_i:
                st.metric("📈 Avg Forecast", f"{recent['yhat'].mean():.0f}", "units")
            with col_ii:
                st.metric("⬆️ Peak", f"{recent['yhat'].max():.0f}", "units")
            with col_iii:
                st.metric("Range Variance", f"±{recent['yhat_upper'].mean() - recent['yhat'].mean():.0f}", "units")
            
            # Forecast chart
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.plot(f['ds'], f['yhat'], label='Forecast', color='#ff6b6b', linewidth=2.5)
            ax.fill_between(f['ds'], f['yhat_lower'], f['yhat_upper'], alpha=0.2, color='#ff6b6b', label='95% CI')
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_xlabel('Date', fontsize=11, color='#1a1a1a')
            ax.set_ylabel('Demand (Units)', fontsize=11, color='#1a1a1a')
            ax.set_title('7-Day Demand Forecast', fontsize=12, fontweight='bold', color='#1a1a1a')
            ax.legend(loc='upper left')
            ax.tick_params(colors='#1a1a1a')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

# ===== TAB 3: CHURN & SUPPLIER =====
with tab3:
    st.markdown("### 👥 Churn & Supplier Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Customer Churn Prediction**")
        file = st.file_uploader("Upload Customer Data (CSV/XLSX)", type=['csv', 'xlsx'], key='churn_upload')
        if file:
            try:
                df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
                st.markdown(f'<div class="success-box">✅ {len(df)} records</div>', unsafe_allow_html=True)
                
                if st.button("🎯 Predict Churn", use_container_width=True):
                    result, msg = run_churn_prediction(df)
                    if result:
                        st.session_state['predictions']['churn'] = result
                        st.rerun()
                    else:
                        st.markdown(f'<div class="error-box">❌ {msg}</div>', unsafe_allow_html=True)
                
                if 'churn' in st.session_state['predictions']:
                    c = st.session_state['predictions']['churn']
                    col_i, col_ii = st.columns(2)
                    col_i.metric("🔴 At-Risk", f"{c['churn']}/{c['total']}", f"{c['pct']:.1f}%")
                    col_ii.metric("🟢 Stable", f"{c['total']-c['churn']}", f"{100-c['pct']:.1f}%")
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ {str(e)}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Supplier Reliability**")
        file = st.file_uploader("Upload Supplier Data (CSV/XLSX)", type=['csv', 'xlsx'], key='supplier_upload')
        if file:
            try:
                df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
                st.markdown(f'<div class="success-box">✅ {len(df)} records</div>', unsafe_allow_html=True)
                
                if st.button("⭐ Score Suppliers", use_container_width=True):
                    result, msg = run_supplier_prediction(df)
                    if result:
                        st.session_state['predictions']['supplier'] = result
                        st.rerun()
                    else:
                        st.markdown(f'<div class="error-box">❌ {msg}</div>', unsafe_allow_html=True)
                
                if 'supplier' in st.session_state['predictions']:
                    s = st.session_state['predictions']['supplier']
                    col_i, col_ii, col_iii = st.columns(3)
                    col_i.metric("📊 Avg Score", f"{s['avg']:.2f}", "/10")
                    col_ii.metric("⭐ Best", f"{s['max']:.2f}", "/10")
                    col_iii.metric("⚠️ Worst", f"{s['min']:.2f}", "/10")
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ {str(e)}</div>', unsafe_allow_html=True)

# ===== TAB 4: SUPPLY CHAIN CHATBOT =====
with tab4:
    st.markdown("### 💬 Supply Chain & Delivery Chatbot")
    st.markdown('<div class="info-box">🎯 Ask me about supply chain, delivery, logistics, or inventory topics!</div>', unsafe_allow_html=True)
    
    user_input = st.text_input("Your question:", placeholder="E.g., 'How to reduce delivery costs?', 'What is supply chain?', 'Delivery optimization tips'")
    
    if user_input:
        if check_supply_chain_relevance(user_input):
            response = get_supply_chain_response(user_input)
            st.session_state['chat_history'].append({"role": "user", "content": user_input})
            st.session_state['chat_history'].append({"role": "bot", "content": response})
            st.rerun()
        else:
            st.session_state['chat_history'].append({"role": "user", "content": user_input})
            st.session_state['chat_history'].append({
                "role": "reject",
                "content": "❌ I can only answer supply chain and delivery-related questions. Please ask about logistics, shipping, inventory, demand forecasting, or similar supply chain topics."
            })
            st.rerun()
    
    st.markdown("---")
    st.markdown("**Chat History:**")
    
    if st.session_state['chat_history']:
        for msg in reversed(st.session_state['chat_history'][-10:]):
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">👤 <b>You:</b><br/>{msg["content"]}</div>', unsafe_allow_html=True)
            elif msg["role"] == "bot":
                st.markdown(f'<div class="chat-bot">🤖 <b>AI:</b><br/>{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-reject">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.info("💭 No chat history yet. Ask a question to get started!")

# ===== TAB 5: INFO =====
with tab5:
    st.markdown("### ℹ️ About This System")
    
    st.markdown("""
    **🏭 Supply Chain AI System - Production Ready**
    
    **Modules:**
    - 📦 **Delivery Prediction** - Risk & Delay forecasting using Random Forest
    - 📈 **Demand Forecasting** - 7-day predictions using Prophet time-series
    - 👥 **Churn & Supplier** - Customer retention and supplier scoring
    - 💬 **Chatbot** - Supply chain domain knowledge base
    
    **Models Used:**
    - Random Forest Classifier: `rf_classifier_pipeline.pkl`
    - Random Forest Regressor: `rf_regressor_pipeline.pkl`
    - Prophet: Time-series forecasting
    - Optional: CatBoost models (*.pkl)
    
    **Required Files:**
    - `rf_classifier_pipeline.pkl` - For delivery risk classification
    - `rf_regressor_pipeline.pkl` - For delay regression
    - Other optional model files in project root
    
    **Technologies:**
    - Streamlit (Web UI)
    - Scikit-learn (Random Forest)
    - Prophet (Forecasting)
    - Pandas & NumPy (Data processing)
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 Configuration")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("**Models Found:**")
        for name, model in models.items():
            st.markdown(f"✅ `{name}`")
        if not models:
            st.markdown("❌ No models found")
    
    with col_info2:
        st.markdown("**Python Requirements:**")
        st.code("""streamlit
pandas
scikit-learn
prophet
joblib
numpy
matplotlib""", language="text")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #ff6b6b; margin-top: 50px;'><p>🏭 <b>Supply Chain AI v9.0</b> | Random Forest + Prophet | Production Ready</p></div>", unsafe_allow_html=True)
