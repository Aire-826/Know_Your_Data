import streamlit as st
import pandas as pd
import plotly.express as px
from model_engine import load_and_preprocess_data, run_automl_benchmark, predict_future_client_revenue

# Page Configuration
st.set_page_config(
    page_title="MLPredict AI", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SuperAdmin-Inspired CSS Styling
st.markdown("""
    <style>
    .main {
        background-color: #0B0F19;
    }
    /* Deep Wine / Maroon Sidebar Styling matching reference */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2D1319 0%, #131B2E 100%) !important;
        border-right: 1px solid #3F1D26;
    }
    [data-testid="stSidebar"] * {
        color: #F3F4F6 !important;
    }
    /* Coral-Red Gradient Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #F43F5E 0%, #BE123C 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 4px 14px rgba(244, 63, 94, 0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(244, 63, 94, 0.6);
    }
    /* Glassmorphism KPI & Container Cards */
    .metric-card {
        background: #1A2234;
        border: 1px solid #4C1D24;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .metric-value {
        color: #F43F5E;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 6px;
    }
    /* Executive Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #2D1319 0%, #131B2E 100%);
        border: 1px solid #5C1D2A;
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
    }
    h1, h2, h3 {
        color: #F8FAFC !important;
        font-family: 'Inter', sans-serif;
    }
    /* Center alignment helper class */
    .center-content {
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

# Executive Hero Banner
st.markdown("""
    <div class="hero-banner center-content">
        <h1 style="color: #FB7185 !important; margin: 0; font-size: 2.4rem;">⚡ MLPredict AI</h1>
        <p style="color: #94A3B8; margin: 5px 0 0 0; font-size: 1.1rem;">Next-Generation Client Revenue & Sales Forecasting Engine for XYZ Electricals</p>
    </div>
""", unsafe_allow_html=True)

# --- HOME PAGE CSV UPLOADER SECTION ---
st.markdown("""
    <div class="center-content">
        <p style="color: #94A3B8; max-width: 600px;">
            Upload your enterprise client sales CSV file below (upto 4 years).
        </p>
    </div>
""", unsafe_allow_html=True)

# Centered layout columns to perfectly center the uploader and status box
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    uploaded_file = st.file_uploader("Upload client sales CSV record (Max 200 MB per file)", type=["csv"])

    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        df_raw.to_csv("client_sales_data.csv", index=False)
        st.success("✅ Custom CSV Synchronized!")
    else:
        try:
            df_raw = pd.read_csv("client_sales_data.csv")
            st.info("🔗 **Status:** Connected to Default Dataset")
        except FileNotFoundError:
            st.error("⚠️ Dataset missing. Run `generate_csvdata.py` first.")
            st.stop()


# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["📊 Executive Dashboard", "🤖 Auto-ML Model", "🔮 Client Revenue Forecaster"])

# --- TAB 1: EXECUTIVE DASHBOARD ---
with tab1:
    st.subheader("Global Enterprise Performance (2023 - 2026)")
    
    total_rev = df_raw['total_revenue'].sum()
    total_units = df_raw['units_sold'].sum()
    unique_clients = df_raw['client_id'].nunique()
    
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">💰 Total Verified Revenue</div>
                <div class="metric-value">₹ {total_rev:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">📦 Volume Shipped</div>
                <div class="metric-value">{total_units:,} Units</div>
            </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">🏢 Active Enterprise Accounts</div>
                <div class="metric-value">{unique_clients} Clients</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    monthly_trend = df_raw.groupby('date')['total_revenue'].sum().reset_index()
    
    # Dual-tone neon line chart matching reference visual style
    fig_trend = px.line(
        monthly_trend, x='date', y='total_revenue',
        title="<b>Long-Term Enterprise Revenue Trajectory</b>",
        labels={'total_revenue': 'Revenue (INR)', 'date': 'Timeline'},
        markers=True,
        template="plotly_dark"
    )
    fig_trend.update_traces(line=dict(color='#F43F5E', width=3.5))
    fig_trend.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color="#F3F4F6")
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    
    c_left, c_right = st.columns(2)
    with c_left:
        category_rev = df_raw.groupby('product_category')['total_revenue'].sum().reset_index()
        fig_cat = px.pie(
            category_rev, names='product_category', values='total_revenue',
            title="<b>Revenue Distribution by Product Line</b>",
            hole=0.55, template="plotly_dark",
            color_discrete_sequence=['#F43F5E', '#38BDF8', '#FB7185']
        )
        fig_cat.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color="#F3F4F6")
        )
        st.plotly_chart(fig_cat, use_container_width=True)
        
    with c_right:
        client_rev = df_raw.groupby('client_id')['total_revenue'].sum().reset_index()
        fig_client = px.bar(
            client_rev, x='client_id', y='total_revenue',
            title="<b>Revenue Share per Enterprise Client</b>",
            labels={'total_revenue': 'Revenue (INR)', 'client_id': 'Client Account'},
            template="plotly_dark", color='client_id',
            color_discrete_sequence=['#F43F5E', '#38BDF8', '#818CF8', '#FB7185', '#34D399']
        )
        fig_client.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)', 
            showlegend=False,
            font=dict(family="Inter", color="#F3F4F6")
        )
        st.plotly_chart(fig_client, use_container_width=True)

# --- TAB 2: AUTO-ML BENCHMARK ---
with tab2:
    st.subheader("Automated Supervised Learning Benchmark Tournament")
    st.markdown("Our automated pipeline executes cross-validation scoring across multiple regression algorithms to isolate the model with peak prediction performance.")
    
    if st.button("🚀 Initialize Auto-ML Tournament"):
        with st.spinner("Executing model training, cross-validation, and performance evaluation..."):
            df_processed, encoders = load_and_preprocess_data("client_sales_data.csv")
            automl_results = run_automl_benchmark(df_processed, target_col='total_revenue')
            
            st.success("Tournament evaluation complete!")
            winner = automl_results['best_algorithm']
            
            st.markdown(f"""
                <div style="background-color: #1A2234; border-left: 5px solid #F43F5E; padding: 15px; border-radius: 8px; margin: 15px 0; border: 1px solid #4C1D24;">
                    <h4 style="margin:0; color: #FB7185;">🏆 Optimal Production Model Selected: {winner}</h4>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### Model Comparison Scoreboard")
            results_df = pd.DataFrame(automl_results['benchmark_results']).T.reset_index()
            results_df.columns = ["Supervised Algorithm", "Mean R² Score", "Standard Deviation"]
            st.dataframe(results_df.sort_values(by="Mean R² Score", ascending=False), use_container_width=True)
            
            st.session_state['trained_models'] = automl_results['trained_models']
            st.session_state['encoders'] = encoders
            st.session_state['best_model_name'] = winner

# --- TAB 3: CLIENT-LEVEL PREDICTIONS ---
with tab3:
    st.subheader("🔮 Predictive Revenue Forecasting Engine")
    st.markdown("Simulate forward-looking purchase metrics for specific enterprise clients under custom operational parameters.")
    
    if 'trained_models' not in st.session_state:
        st.warning("⚠️ Please execute the Auto-ML Model Tournament in Tab 2 first to load trained models into active session memory.")
    else:
        clients_list = df_raw['client_id'].unique().tolist()
        categories_list = df_raw['product_category'].unique().tolist()
        
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            selected_client = st.selectbox("Select Enterprise Client", clients_list)
            selected_category = st.selectbox("Select Product Line", categories_list)
            target_year = st.selectbox("Target Projection Year", [2027, 2028, 2029])
            
        with col_in2:
            target_month = st.slider("Target Projection Month", min_value=1, max_value=12, value=6)
            input_price = st.number_input("Projected Unit Price (INR)", min_value=100.0, value=500.0, step=50.0)
            input_units = st.number_input("Projected Unit Volume Demand", min_value=1, value=150, step=10)
            
        if st.button("Generate Revenue Forecast"):
            best_model = st.session_state['trained_models'][st.session_state['best_model_name']]
            encoders = st.session_state['encoders']
            
            predicted_rev = predict_future_client_revenue(
                model=best_model,
                encoders=encoders,
                client_id=selected_client,
                category=selected_category,
                unit_price=input_price,
                units_sold=input_units,
                target_year=target_year,
                target_month=target_month
            )
            
            st.markdown("---")
            st.markdown(f"""
                <div style="background-color: #1A2234; border: 1px solid #4C1D24; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #FB7185; margin: 0;">📈 Forecasted Revenue for {selected_client}</h3>
                    <h1 style="color: #F43F5E; margin: 10px 0 0 0;">₹ {predicted_rev:,.2f}</h1>
                    <p style="color: #94A3B8; margin: 5px 0 0 0; font-size: 0.9rem;">Generated via <b>{st.session_state['best_model_name']}</b> inference engine.</p>
                </div>
            """, unsafe_allow_html=True)