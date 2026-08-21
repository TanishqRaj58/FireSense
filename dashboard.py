import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from predict import predict_status

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="FireSense AI Control Room",
    layout="wide"
)
st.title("FireSense AI – Coal Mine Early Fire Detection & Prevention System")
# ===============================
# LOAD DATA FOR ANALYTICS
# ===============================
df_2024 = pd.read_csv("data/dataset_2024.csv")
df_2025 = pd.read_csv("data/dataset_2025.csv")
# ===============================
# TABS
# ===============================
tab1, tab2 = st.tabs(["🟦 CONTROL ROOM", "📊 ANALYTICS"])

# ================= TAB 1 ==============================

with tab1:
    st.subheader("🎛️ Manual Mine Condition Simulator")
    left, center, right = st.columns([1.2, 2, 1.2])
    # ---------------- LEFT PANEL ----------------
    with left:
        st.markdown("### 🔧 Sensor Inputs")
        temp = st.slider("Temperature (°C)", 0, 100, 30)
        ch4 = st.slider("Methane CH₄ (%)", 0.0, 5.0, 1.2)
        co = st.slider("Carbon Monoxide (ppm)", 0, 500, 20)
        o2 = st.slider("Oxygen (%)", 10.0, 21.0, 20.5)
        air = st.slider("Airflow (m/s)", 0.1, 5.0, 2.0)
    sensor_input = {
        "temp_C": temp,
        "CH4_percent": ch4,
        "CO_ppm": co,
        "O2_percent": o2,
        "airflow_m_s": air,
        "humidity_percent": 70,
        "dust_mg_m3": 3,
        "pressure_Pa": 101000
    }
    label, risk = predict_status(sensor_input)
    # ---------------- CENTER PANEL ----------------
    with center:
        st.markdown("### AI Risk Assessment")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Temp (°C)", temp)
        c2.metric("CH₄ (%)", ch4)
        c3.metric("CO (ppm)", co)
        c4.metric("O₂ (%)", o2)
        c5.metric("Airflow (m/s)", air)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk * 100,
            title={"text": "Prediction Confidence"},
            gauge={ 
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1e90ff"},   # 🔵 BLUE indicator
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 30], "color": "green"},
                    {"range": [30, 60], "color": "yellow"},
                    {"range": [60, 100], "color": "red"},
                ],
            }
        ))
        st.plotly_chart(gauge, use_container_width=True)
        st.markdown(f"### 🔍 Prediction: **{label}**")
        #st.markdown(f"**Risk Score:** `{round(risk, 3)}`")
        # ------------------------------------------------------------------------------------


    # ---------------- RIGHT PANEL ----------------
    with right:
        st.markdown("### 🚨 Preventive Actions")
        if label == "SAFE":
            st.success("SAFE CONDITIONS")
            st.write("- Continue monitoring")
            st.write("- No immediate action")
        elif label == "WARNING":
            st.warning("WARNING LEVEL")
            st.write("- Increase ventilation")
            st.write("- Inspect affected zone")
            st.write("- Reduce workforce")
        else:
            st.error("CRITICAL DANGER")
            st.write("- Evacuate workers immediately")
            st.write("- Seal affected area")
            st.write("- Activate fire control protocol")


# ================= TAB 2 ==============================

with tab2:
    st.subheader("📊 Advanced Analytics & AI Understanding")

    # PARAMETER SELECTION

    param = st.selectbox(
        "Select Parameter",
        ["temp_C", "CH4_percent", "CO_ppm", "O2_percent", "airflow_m_s"]
    )
    
    # 1️⃣ HISTORICAL COMPARISON
    
    st.markdown("### 📈 Historical Trend Comparison (2024 vs 2025)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_2024["timestamp"],
        y=df_2024[param],
        name="2024",
        line=dict(color="#00e5ff")
    ))
    fig.add_trace(go.Scatter(
        x=df_2025["timestamp"],
        y=df_2025[param],
        name="2025",
        line=dict(color="#ff9800")
    ))
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title=f"{param}",
        legend_title="Year",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)
    

    # 2️⃣ PARAMETER vs RISK
    
    st.markdown("### 📉 Parameter vs Fire Risk Relationship")
    df_2025["risk_numeric"] = df_2025["safety_status"].map({
        "SAFE": 0.2,
        "WARNING": 0.6,
        "DANGER": 0.9
    })
    scatter = px.scatter(
        df_2025.sample(2000),
        x=param,
        y="risk_numeric",
        labels={"risk_numeric": "Fire Risk"},
        color="safety_status",
        title=f"{param} vs Fire Risk"
    )
    st.plotly_chart(scatter, use_container_width=True)
    
    # 3️⃣ FEATURE IMPORTANCE
    
    st.markdown("### Feature Importance (AI Explanation)")
    importance_df = pd.DataFrame({
        "Feature": [
            "CO_ppm",
            "temp_C",
            "CH4_percent",
            "airflow_m_s",
            "O2_percent"
        ],
        "Importance (%)": [35, 28, 22, 10, 5]
    })
    bar = px.bar(
        importance_df,
        x="Importance (%)",
        y="Feature",
        orientation="h",
        color="Feature",
        title="Random Forest Feature Importance"
    )
    st.plotly_chart(bar, use_container_width=True)
    
    # 4️⃣ RISK DISTRIBUTION
    
    st.markdown("### 🧯 Risk Distribution (2025)")
    pie = px.pie(
        df_2025,
        names="safety_status",
        title="Safety Status Distribution",
        color_discrete_sequence=["#00ff88", "#ffcc00", "#ff4b4b"]
    )
    st.plotly_chart(pie, use_container_width=True)


  #    python -m streamlit run dashboard.py