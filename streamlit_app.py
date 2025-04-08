import streamlit as st

# --- Page configuration ---
st.set_page_config(page_title="🧠 AI Insight Suite", page_icon="🧠", layout="centered")

# --- Hide sidebar initially (only show in sub-apps) ---
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title and Intro ---
st.title("🧠 AI Insight Suite")
st.markdown("""
Welcome to your unified AI Assistant platform.  
This suite combines powerful business analytics and customer feedback insights into one seamless tool.
""")

# --- Feature Descriptions ---
st.markdown("### 🚀 What You Can Do Here:")

st.markdown("""
- 📈 **Business Analysis Assistant**  
  Generate structured insights from business problems and supporting CSV data.  
  Identify requirements, evaluate solutions, and get action plans.

- 💬 **Customer Feedback Analyzer**  
  Upload feedback (from surveys, reviews, etc.) and let the AI extract sentiment, categorize issues, and answer questions.
""")

st.markdown("---")

# --- Navigation Dropdown ---
tool_choice = st.selectbox("🔐 Choose a Tool to Get Started:", ["Select...", "📈 Business Analysis Assistant", "💬 Customer Feedback Analyzer"])

# --- Redirect Based on Selection ---
if tool_choice == "📈 Business Analysis Assistant":
    st.switch_page("pages/1_Business_Analysis.py")
elif tool_choice == "💬 Customer Feedback Analyzer":
    st.switch_page("pages/2_Feedback_Analyzer.py")
 