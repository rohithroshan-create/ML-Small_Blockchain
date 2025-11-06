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

# CSS
st.markdown("""
    <style>
    * { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .main { background: #f8f9fa; color: #1a1a1a; }
    .header-box { background: linear-gradient(135deg, #ff6b6b 0%, #c44569 100%); color: white; padding: 30px; border-radius: 15px; margin: 20px 0; }
    .header-box h1 { color: white; margin: 0; }
    .success-box { background: #4caf50; color: white; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid white; font-weight: bold; }
    .error-box { background: #ff6b6b; color: white; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid white; font-weight: bold; }
    .chat-user { background: #2196F3; color: white; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid white; }
    .chat-bot { background: #4caf50; color: white; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid white; }
    .chat-reject { background: #ff9800; color: white; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid white; }
    </style>
""", unsafe_allow_html=True)

# SESSION STATE
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = {}
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# LOAD MODELS
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

# PREDICTION FUNCTIONS
def run_delivery_risk(df):
    try:
        if 'delivery_risk' not in models:
            return None, "Model not found"
        cols = ['Days for shipping (real)', 'Days for shipment (scheduled)', 'Shipping Mode', 'Order Item Quantity']
        if not all(col in df.columns for col in cols):
            return None, "Missing columns"
        pred = models['delivery_risk'].predict(df[cols])
        prob = models['delivery_risk'].predict_proba(df[cols])
        return {'pred': pred, 'prob': prob, 'risk': sum(pred), 'total': len(pred), 'pct': (sum(pred)/len(pred)*100)}, "OK"
    except Exception as e:
        return None, str(e)

def run_delay(df):
    try:
        if 'delay_regression' not in models:
            return None, "Model not found"
        cols = ['Days for shipping (real)', 'Days for shipment (scheduled)', 'Shipping Mode', 'Order Item Quantity']
        if not all(col in df.columns for col in cols):
            return None, "Missing columns"
        pred = models['delay_regression'].predict(df[cols])
        return {'pred': pred, 'avg': np.mean(pred), 'max': np.max(pred), 'min': np.min(pred)}, "OK"
    except Exception as e:
        return None, str(e)

def run_churn(df):
    try:
        if 'churn' not in models:
            return None, "Model not found"
        cols = ['Customer Segment', 'Type', 'Category Name', 'Order Item Quantity', 'Sales', 'Order Profit Per Order']
        if not all(col in df.columns for col in cols):
            return None, "Missing columns"
        pred = models['churn'].predict(df[cols])
        prob = models['churn'].predict_proba(df[cols])
        return {'pred': pred, 'prob': prob, 'churn': sum(pred), 'total': len(pred), 'pct': (sum(pred)/len(pred)*100)}, "OK"
    except Exception as e:
        return None, str(e)

def run_supplier(df):
    try:
        if 'supplier' not in models:
            return None, "Model not found"
        cols = ['Days for shipping (real)', 'Days for shipment (scheduled)', 'Shipping Mode', 'Order Item Quantity']
        if not all(col in df.columns for col in cols):
            return None, "Missing columns"
        pred = models['supplier'].predict(df[cols])
        return {'pred': pred, 'avg': np.mean(pred), 'max': np.max(pred), 'min': np.min(pred)}, "OK"
    except Exception as e:
        return None, str(e)

def check_supply_chain_relevance(question):
    """Check if question is supply chain/delivery related"""
    keywords = ['delivery', 'delay', 'supply', 'chain', 'shipping', 'shipment', 'order', 'customer', 'churn', 
                'supplier', 'demand', 'forecast', 'logistics', 'warehouse', 'inventory', 'transport', 'carrier',
                'packaging', 'route', 'cost', 'time', 'performance', 'risk', 'optimization', 'planning',
                'tips', 'how to', 'improve', 'reduce', 'increase', 'best practice', 'management']
    return any(kw in question.lower() for kw in keywords)

def get_supply_chain_response(question):
    """Provide supply chain/delivery domain responses"""
    q = question.lower()
    
    if 'delivery' in q and ('how' in q or 'tip' in q or 'improve' in q):
        return """**Supply Chain Delivery Tips:**

1. **Route Optimization**
   - Use GPS-based routing for fastest paths
   - Consolidate orders by delivery zones
   - Reduce per-order delivery cost by 20-30%

2. **Carrier Selection**
   - Express for urgent orders (2-3 days)
   - Standard for regular orders (5-7 days)
   - Compare carrier rates quarterly

3. **Packaging**
   - Lightweight packaging reduces costs
   - Proper packaging reduces damage claims
   - Eco-friendly options improve brand image

4. **Timing**
   - Ship early in the week for better delivery
   - Avoid peak seasons when possible
   - Plan for holiday delays"""
    
    elif 'reduce delay' in q or 'faster delivery' in q:
        return """**Ways to Reduce Delivery Delays:**

1. **Process Improvements**
   - Streamline order picking (batch by area)
   - Pre-pack common orders
   - Automate labeling and sorting

2. **Supplier Coordination**
   - Set clear SLAs with suppliers
   - Daily communication with logistics partners
   - Monitor supplier performance

3. **Technology**
   - Real-time tracking system
   - Automated alerts for delays
   - Predictive analytics for risk orders

4. **Network Optimization**
   - Regional distribution centers
   - Local warehouses for fast-moving items
   - Partner with multiple carriers"""
    
    elif 'supply chain' in q and ('what' in q or 'explain' in q):
        return """**Supply Chain Overview:**

Supply Chain includes:
1. **Procurement** - Sourcing materials from suppliers
2. **Production** - Manufacturing products
3. **Distribution** - Moving goods to warehouses
4. **Logistics** - Shipping to customers
5. **Returns** - Processing returns and refunds

**Key Metrics:**
- Lead time (supplier to warehouse)
- Fulfillment time (order to shipment)
- Delivery time (shipment to customer)
- Cost per unit
- Customer satisfaction"""
    
    elif 'cost' in q or 'expensive' in q:
        return """**Reducing Supply Chain Costs:**

1. **Shipping Optimization**
   - Negotiate bulk rates (5-15% savings)
   - Use slower shipping when acceptable
   - Consolidate shipments

2. **Inventory Management**
   - Reduce excess stock (less storage cost)
   - Improve demand forecasting
   - Use just-in-time delivery

3. **Supplier Management**
   - Develop relationships for better pricing
   - Consolidate suppliers (reduce variety)
   - Source from regional suppliers

4. **Process Efficiency**
   - Automate order picking
   - Reduce manual sorting
   - Improve warehouse layout"""
    
    elif 'demand' in q and ('forecast' in q or 'predict' in q):
        return """**Demand Forecasting in Supply Chain:**

Purpose: Predict customer demand to optimize inventory

**Methods:**
1. Historical data analysis
2. Seasonal adjustments
3. Trend analysis
4. Prophet time-series forecasting

**Benefits:**
- Reduce excess inventory
- Prevent stockouts
- Optimize warehouse space
- Better supplier planning

**Use this app's Demand Module to forecast!**"""
    
    elif 'customer' in q and ('retention' in q or 'loyalty' in q):
        return """**Customer Retention Tips:**

1. **Communication**
   - Track order status in real-time
   - Proactive delay notifications
   - Dedicated customer support

2. **Quality**
   - On-time delivery consistently
   - Proper packaging prevents damage
   - Easy returns process

3. **Incentives**
   - Loyalty programs
   - Bulk order discounts
   - Free shipping thresholds

4. **Feedback**
   - Survey customers regularly
   - Act on feedback quickly
   - Show improvement"""
    
    elif 'supplier' in q and ('choose' in q or 'select' in q or 'evaluate' in q):
        return """**Choosing Suppliers:**

**Key Criteria:**
1. Reliability (on-time delivery %)
2. Quality (defect rate)
3. Cost competitiveness
4. Communication responsiveness
5. Scalability for growth
6. Location/proximity
7. Financial stability

**Evaluation Process:**
- Request quotes
- Check references
- Review performance metrics
- Trial orders first
- Establish SLAs
- Regular performance reviews"""
    
    elif 'warehouse' in q or 'storage' in q or 'inventory' in q:
        return """**Warehouse & Inventory Management:**

**Inventory Optimization:**
1. ABC analysis (classify by value)
2. Keep fast-movers readily accessible
3. Implement FIFO (first in, first out)
4. Regular stock audits

**Warehouse Organization:**
- Organize by product type
- Clear labeling system
- Efficient picking routes
- Safety protocols

**Technology:**
- Inventory management system
- Barcode tracking
- Real-time stock visibility
- Automated alerts for low stock"""
    
    elif 'return' in q or 'reverse' in q:
        return """**Returns & Reverse Logistics:**

**Return Process:**
1. Accept return within reasonable period
2. Inspect item condition
3. Restock or dispose appropriately
4. Process refund

**Optimization:**
- Minimize return rate through quality
- Establish return hubs
- Partner carriers for pickups
- Track return metrics

**Best Practices:**
- Easy return process
- Fast refunds (within 5-7 days)
- Environmental considerations
- Data collection on return reasons"""
    
    else:
        return """**Supply Chain Questions I Can Help With:**

- Delivery optimization tips
- Reducing costs
- Supply chain overview
- Demand forecasting
- Warehouse management
- Supplier selection
- Customer retention
- Reverse logistics

Ask me about any supply chain/delivery topic!"""

# HEADER
st.markdown('<div class="header-box"><h1>🏭 Supply Chain AI System</h1></div>', unsafe_allow_html=True)

# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📦 Delivery", "📈 Demand", "👥 Churn & Supplier", "💬 Supply Chain Chatbot", "ℹ️ Info"])

# ===== TAB 1: DELIVERY =====
with tab1:
    st.markdown("### 📦 Delivery Risk & Delay Prediction")
    
    col_upload, col_results = st.columns([1, 1.3])
    
    with col_upload:
        st.markdown("**Step 1: Upload Data**")
        file = st.file_uploader("Upload CSV/XLSX", type=['csv', 'xlsx'], key='delivery_upload')
        if file:
            df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
            st.markdown(f'<div class="success-box">✅ {len(df)} records loaded</div>', unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔴 Risk Prediction", use_container_width=True):
                    result, msg = run_delivery_risk(df)
                    if result:
                        st.session_state['predictions']['delivery_risk'] = result
                        st.rerun()
            with col_b:
                if st.button("⏱️ Delay Prediction", use_container_width=True):
                    result, msg = run_delay(df)
                    if result:
                        st.session_state['predictions']['delay'] = result
                        st.rerun()
    
    with col_results:
        st.markdown("**Results**")
        if 'delivery_risk' in st.session_state['predictions']:
            r = st.session_state['predictions']['delivery_risk']
            col_i, col_ii, col_iii = st.columns(3)
            col_i.metric("🔴 High Risk", r['risk'], f"{r['pct']:.1f}%")
            col_ii.metric("🟢 On-Time", r['total']-r['risk'], f"{100-r['pct']:.1f}%")
            col_iii.metric("📦 Total", r['total'], "100%")
            
            st.dataframe({
                'Order': range(min(5, len(r['pred']))),
                'Status': ['🔴 HIGH' if r['pred'][i]==1 else '🟢 ON-TIME' for i in range(min(5, len(r['pred'])))],
                'Confidence': [f"{max(r['prob'][i])*100:.1f}%" for i in range(min(5, len(r['pred'])))]
            }, use_container_width=True, hide_index=True)
        
        if 'delay' in st.session_state['predictions']:
            d = st.session_state['predictions']['delay']
            col_i, col_ii, col_iii = st.columns(3)
            col_i.metric("📊 Avg", f"{d['avg']:.2f}", "days")
            col_ii.metric("⬆️ Max", f"{d['max']:.2f}", "days")
            col_iii.metric("⬇️ Min", f"{d['min']:.2f}", "days")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(d['pred'], bins=15, color='#ff6b6b', edgecolor='white')
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_xlabel('Delay (Days)')
            ax.set_ylabel('Frequency')
            ax.set_title('Delay Distribution')
            plt.tight_layout()
            st.pyplot(fig)

# ===== TAB 2: DEMAND =====
with tab2:
    st.markdown("### 📈 Demand Forecasting (Prophet)")
    
    col_upload, col_results = st.columns([1, 1.3])
    
    with col_upload:
        st.markdown("**Step 1: Upload Data**")
        file = st.file_uploader("Upload CSV/XLSX (date, sales)", type=['csv', 'xlsx'], key='demand_upload')
        if file:
            df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
            st.markdown(f'<div class="success-box">✅ {len(df)} records loaded</div>', unsafe_allow_html=True)
            
            if st.button("🔮 Forecast 7 Days", use_container_width=True):
                try:
                    df['date'] = pd.to_datetime(df['date'])
                    day_sales = df.groupby('date')['sales'].sum().reset_index()
                    pdf = day_sales.rename(columns={"date": "ds", "sales": "y"})
                    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                    model.fit(pdf)
                    future = model.make_future_dataframe(periods=7)
                    forecast = model.predict(future)
                    st.session_state['predictions']['prophet'] = forecast
                    st.rerun()
                except Exception as e:
                    st.markdown(f'<div class="error-box">❌ {str(e)}</div>', unsafe_allow_html=True)
    
    with col_results:
        st.markdown("**Results**")
        if 'prophet' in st.session_state['predictions']:
            f = st.session_state['predictions']['prophet']
            recent = f.tail(7)
            col_i, col_ii, col_iii = st.columns(3)
            col_i.metric("📈 Avg", f"{recent['yhat'].mean():.0f}", "units")
            col_ii.metric("⬆️ Peak", f"{recent['yhat'].max():.0f}", "units")
            col_iii.metric("Range", f"±{recent['yhat_upper'].mean()-recent['yhat'].mean():.0f}", "units")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(f['ds'], f['yhat'], label='Forecast', color='#ff6b6b', linewidth=2)
            ax.fill_between(f['ds'], f['yhat_lower'], f['yhat_upper'], alpha=0.2, color='#ff6b6b')
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_xlabel('Date')
            ax.set_ylabel('Demand')
            ax.set_title('7-Day Demand Forecast')
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

# ===== TAB 3: CHURN & SUPPLIER =====
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Churn Prediction**")
        file = st.file_uploader("Upload Customer Data", type=['csv', 'xlsx'], key='churn_upload')
        if file:
            df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
            st.markdown(f'<div class="success-box">✅ {len(df)} records</div>', unsafe_allow_html=True)
            if st.button("🎯 Predict", use_container_width=True):
                result, _ = run_churn(df)
                if result:
                    st.session_state['predictions']['churn'] = result
                    st.rerun()
            if 'churn' in st.session_state['predictions']:
                c = st.session_state['predictions']['churn']
                st.metric("At-Risk", f"{c['churn']}/{c['total']}", f"{c['pct']:.1f}%")
    
    with col2:
        st.markdown("**Supplier Reliability**")
        file = st.file_uploader("Upload Supplier Data", type=['csv', 'xlsx'], key='supplier_upload')
        if file:
            df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
            st.markdown(f'<div class="success-box">✅ {len(df)} records</div>', unsafe_allow_html=True)
            if st.button("⭐ Score", use_container_width=True):
                result, _ = run_supplier(df)
                if result:
                    st.session_state['predictions']['supplier'] = result
                    st.rerun()
            if 'supplier' in st.session_state['predictions']:
                s = st.session_state['predictions']['supplier']
                st.metric("Avg Score", f"{s['avg']:.2f}", "/10")

# ===== TAB 4: SUPPLY CHAIN CHATBOT =====
with tab4:
    st.markdown("### 💬 Supply Chain & Delivery Chatbot")
    st.info("🎯 Ask me about supply chain, delivery, logistics, or inventory topics!")
    
    user_input = st.text_input("Your question:", placeholder="E.g., 'How to reduce delivery costs?', 'What is supply chain?'")
    
    if user_input:
        if check_supply_chain_relevance(user_input):
            response = get_supply_chain_response(user_input)
            st.session_state['chat_history'].append({"role": "user", "content": user_input})
            st.session_state['chat_history'].append({"role": "bot", "content": response})
            st.rerun()
        else:
            st.session_state['chat_history'].append({"role": "user", "content": user_input})
            st.session_state['chat_history'].append({"role": "reject", "content": "❌ Sorry, I can only answer supply chain and delivery-related questions. Please ask about logistics, shipping, inventory, demand forecasting, or similar topics."})
            st.rerun()
    
    st.markdown("---")
    for msg in st.session_state['chat_history']:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">👤 <b>You:</b><br/>{msg["content"]}</div>', unsafe_allow_html=True)
        elif msg["role"] == "bot":
            st.markdown(f'<div class="chat-bot">🤖 <b>Bot:</b><br/>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-reject">{msg["content"]}</div>', unsafe_allow_html=True)

# ===== TAB 5: INFO =====
with tab5:
    st.markdown("### ℹ️ About This App")
    st.markdown("""
    **Supply Chain AI System** - Production Ready
    
    **Modules:**
    - 📦 Delivery Risk & Delay Prediction
    - 📈 Demand Forecasting (Prophet)
    - 👥 Customer Churn & Supplier Reliability
    - 💬 Supply Chain Chatbot
    
    **Models Used:**
    - CatBoost for classification & regression
    - Prophet for time-series forecasting
    
    **Requirements:**
    - Model files (*.pkl) in project root
    - Python 3.8+
    - Streamlit, Pandas, Scikit-learn, Prophet
    """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #ff6b6b;'><p>🏭 Supply Chain AI v8.0 | Production Ready</p></div>", unsafe_allow_html=True)
