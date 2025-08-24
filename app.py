# ===================== FLIGHT DASHBOARD (AI + Charts + Sections) =====================
import pandas as pd
import numpy as np
from openai import OpenAI
import os
import streamlit as st
import plotly.express as px

# ===================== CONFIG =====================
client = OpenAI(base_url="https://openrouter.ai/api/v1")
api_key = os.getenv("OPENAI_API_KEY")  # Use Streamlit secrets

# ===================== LOAD DATA =====================
df = pd.read_csv("clean_flights.csv")
for col in ["STD_dt", "ATD_dt", "STA_dt", "ATA_dt"]:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Calculate delays
df["TakeoffDelay_min"] = (df["ATD_dt"] - df["STD_dt"]).dt.total_seconds() / 60
df["LandingDelay_min"] = (df["ATA_dt"] - df["STA_dt"]).dt.total_seconds() / 60
df["TakeoffDelay_min"] = df["TakeoffDelay_min"].fillna(0)
df["LandingDelay_min"] = df["LandingDelay_min"].fillna(0)
df["STD_hour"] = df["STD_dt"].dt.hour
df["STA_hour"] = df["STA_dt"].dt.hour

# Simulate Runway Data if not available
if 'Runway' not in df.columns:
    df['Runway'] = np.random.choice(['R1','R2','R3'], size=len(df))

# ===================== ANALYSIS FUNCTIONS =====================
def best_takeoff_times(plot=False):
    data = df.groupby("STD_hour")["TakeoffDelay_min"].mean().reset_index()
    if plot:
        fig = px.bar(data, x='STD_hour', y='TakeoffDelay_min',
                     title='Average Takeoff Delay by Hour',
                     labels={'STD_hour':'Hour','TakeoffDelay_min':'Delay (min)'},
                     color='TakeoffDelay_min', color_continuous_scale='Viridis')
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        return fig
    return data.set_index("STD_hour")["TakeoffDelay_min"]

def best_landing_times(plot=False):
    data = df.groupby("STA_hour")["LandingDelay_min"].mean().reset_index()
    if plot:
        fig = px.bar(data, x='STA_hour', y='LandingDelay_min',
                     title='Average Landing Delay by Hour',
                     labels={'STA_hour':'Hour','LandingDelay_min':'Delay (min)'},
                     color='LandingDelay_min', color_continuous_scale='Viridis')
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        return fig
    return data.set_index("STA_hour")["LandingDelay_min"]

def busiest_times(plot=False):
    data = df.groupby("STD_hour").size().reset_index(name='Flights')
    if plot:
        fig = px.bar(data, x='STD_hour', y='Flights',
                     title='Busiest Departure Hours',
                     labels={'STD_hour':'Hour','Flights':'Number of Flights'},
                     color='Flights', color_continuous_scale='Viridis')
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        return fig
    return data.set_index("STD_hour")["Flights"]

def tune_schedule_model(plot=False):
    df['tuned_STD_hour'] = (df['STD_hour'] + 1) % 24
    data = df.groupby('tuned_STD_hour')['TakeoffDelay_min'].mean().reset_index()
    if plot:
        fig = px.line(data, x='tuned_STD_hour', y='TakeoffDelay_min', markers=True,
                      title='Effect of Shifting Schedule by 1 Hour',
                      labels={'tuned_STD_hour':'Shifted Hour','TakeoffDelay_min':'Delay (min)'})
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        return fig
    return data.set_index('tuned_STD_hour')["TakeoffDelay_min"]

def cascading_delays(plot=False):
    data = df.groupby('Flight Number')['TakeoffDelay_min'].sum().sort_values(ascending=False).head(10).reset_index()
    if plot:
        fig = px.bar(data, x='Flight Number', y='TakeoffDelay_min',
                     title='Top Flights Causing Cascading Delays',
                     labels={'TakeoffDelay_min':'Total Delay (min)'},
                     color='TakeoffDelay_min', color_continuous_scale='Viridis')
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        return fig
    return data.set_index('Flight Number')["TakeoffDelay_min"]

def overall_summary():
    return {
        'avg_takeoff_delay': df['TakeoffDelay_min'].mean(),
        'avg_landing_delay': df['LandingDelay_min'].mean(),
        'min_takeoff_delay_hour': best_takeoff_times(plot=False).idxmin(),
        'min_landing_delay_hour': best_landing_times(plot=False).idxmin()
    }

# ===================== AI QUERY INTERFACE =====================
def ask_ai(query: str):
    takeoff_info = best_takeoff_times(plot=False).to_dict()
    landing_info = best_landing_times(plot=False).to_dict()
    summary = overall_summary()
    context = f"""
Flight Delay Analysis Insights:
- Average takeoff delay (min): {summary['avg_takeoff_delay']:.2f}
- Average landing delay (min): {summary['avg_landing_delay']:.2f}
- Best takeoff hour: {summary['min_takeoff_delay_hour']}
- Best landing hour: {summary['min_landing_delay_hour']}
- Takeoff delays by hour: {takeoff_info}
- Landing delays by hour: {landing_info}
"""
    response = client.chat.completions.create(
        model='openai/gpt-4o-mini',
        messages=[
            {'role':'system','content':'You are an AI assistant for flight delay analysis.'},
            {'role':'user','content':context},
            {'role':'user','content':query}
        ]
    )
    return response.choices[0].message.content

def handle_query(q):
    q_lower = q.lower()
    if 'takeoff' in q_lower: chart = best_takeoff_times(plot=True)
    elif 'landing' in q_lower: chart = best_landing_times(plot=True)
    elif 'busiest' in q_lower: chart = busiest_times(plot=True)
    elif 'tune' in q_lower: chart = tune_schedule_model(plot=True)
    elif 'cascading' in q_lower: chart = cascading_delays(plot=True)
    else: chart = None
    answer = ask_ai(q)
    return chart, answer

# ===================== STREAMLIT FRONTEND =====================
st.set_page_config(page_title="Flight Dashboard", layout="wide", page_icon="✈")

# ---------------- CSS ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom, #e0f7fa, #ffffff) !important;
}

/* Chatbot input styling - oval */
.stTextInput>div>div>input, .stTextInput>div>div>textarea { 
    color: black !important; 
    background-color: #f0f0f0 !important; 
    border-radius: 25px !important;
    padding: 12px !important;
}

/* AI response card */
.stInfo { 
    border-left:5px solid #4caf50 !important; 
    background-color:#f1f8e9 !important; 
    border-radius:10px !important; 
    padding:15px !important; 
    font-size:16px !important; 
}

/* Menu cards */
.menu-card { display:inline-block; background: linear-gradient(135deg,#4caf50,#81c784); color:white; padding:20px 25px; margin:10px; border-radius:15px; text-align:center; cursor:pointer; font-weight:bold; font-size:16px; box-shadow:2px 2px 12px rgba(0,0,0,0.2); transition:0.4s;}
.menu-card:hover { transform: scale(1.05); box-shadow:4px 4px 20px rgba(0,0,0,0.3); }

h1,h2,h3,h4,h5,h6,p,span,li { color: black !important; font-family: 'Segoe UI', Tahoma, sans-serif; }
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown('<h1 style="text-align:center;">✈ Flight Scheduling & Optimization ✈</h1>', unsafe_allow_html=True)

# ---------------- Chatbot Query ----------------
query = st.text_input("Ask your flight query (e.g., 'Best takeoff time?')")

if st.button("Get AI Answer") and query.strip():
    with st.spinner("AI is analyzing..."):
        chart, ai_response = handle_query(query)
        st.success("✅ AI Response")
        st.info(ai_response)
        if chart is not None:
            st.plotly_chart(chart, use_container_width=True)

# ---------------- Menu Cards ----------------
st.markdown("### 📋 Dashboard Sections")
menu_options = ["🚨 Alerts", "✈ Flight Data", "🛬 Runway Data", "⏱ Current Delays", "⚡ Optimized Schedule"]
clicked_option = st.session_state.get("clicked_option", "")

menu_cols = st.columns(len(menu_options))
for i, option in enumerate(menu_options):
    if menu_cols[i].button(option):
        st.session_state["clicked_option"] = option
        clicked_option = option

# ---------------- Display Cards Function ----------------
def display_card(title, content, chart=None):
    st.markdown(f"### {title}")
    st.markdown(content)
    if chart is not None:
        st.plotly_chart(chart, use_container_width=True)

# ---------------- Content Based on Menu Click ----------------
if clicked_option == "🚨 Alerts":
    takeoff_max = df['TakeoffDelay_min'].max()
    landing_max = df['LandingDelay_min'].max()
    alerts = []
    if takeoff_max > 60: alerts.append(f"⚠ High takeoff delay: {takeoff_max:.1f} min")
    if landing_max > 60: alerts.append(f"⚠ High landing delay: {landing_max:.1f} min")
    if not alerts: alerts.append("✅ No critical alerts!")
    for alert in alerts: display_card("Alert", alert)

elif clicked_option == "✈ Flight Data":
    total = len(df)
    content = f"**Total Flights:** {total}"
    display_card("Flight Data Summary", content, best_takeoff_times(plot=True))

elif clicked_option == "🛬 Runway Data":
    usage = df['Runway'].value_counts().reset_index()
    usage.columns = ['Runway','Flights']
    fig = px.bar(usage, x='Runway', y='Flights', color='Flights', color_continuous_scale='Viridis',
                 title='Flights per Runway')
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    display_card("Runway Data", "Runway usage statistics", fig)

elif clicked_option == "⏱ Current Delays":
    avg_takeoff = df['TakeoffDelay_min'].mean()
    avg_landing = df['LandingDelay_min'].mean()
    content = f"**Average Takeoff Delay:** {avg_takeoff:.2f} min\n**Average Landing Delay:** {avg_landing:.2f} min"
    display_card("Current Delays", content, best_takeoff_times(plot=True))

elif clicked_option == "⚡ Optimized Schedule":
    tuned = tune_schedule_model(plot=True)
    display_card("Optimized Schedule", "Effect of shifting schedule by 1 hour", tuned)
