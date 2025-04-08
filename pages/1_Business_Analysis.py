import streamlit as st
import openai
from dotenv import load_dotenv
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import numpy as np

# --- Load API keys ---
if "OPENAI_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_KEY"]
else:
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Streamlit Config ---
st.set_page_config(page_title="AI Business Analysis Assistant", page_icon="📈")
st.title("📈 AI Business Analysis Assistant")
st.markdown("Paste a business challenge and optionally upload supporting data to generate structured AI insights.")

# --- Input Section ---
st.subheader("📝 Business Input")
user_input = st.text_area("Describe the business challenge, issue, or idea:", height=200)

# --- File Upload ---
st.subheader("📁 Upload Supporting Business Data (CSV)")
uploaded_file = st.file_uploader("Optional: Upload a CSV file to support the analysis", type=["csv"])

df = None
data_summary = ""

try:
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())
        st.subheader("📈 Summary Stats")
        st.write(df.describe())

        chart_options = []
        cat_cols = df.select_dtypes(include="object").columns
        num_cols = df.select_dtypes(include="number").columns

        if len(cat_cols) > 0:
            for col in cat_cols:
                if df[col].nunique() < 15:
                    chart_options.append(f"Bar Chart of {col}")
        if len(num_cols) > 0:
            for col in num_cols:
                chart_options.append(f"Histogram of {col}")
        if len(num_cols) >= 2:
            chart_options.append("Scatter Plot (Pick Two)")
        if len(num_cols) >= 1 and len(cat_cols) >= 1:
            chart_options.append("Faceted Chart by Category")

        if chart_options:
            st.subheader("📈 Visualizations")
            selected_chart = st.selectbox("Choose chart type", chart_options)

            if selected_chart.startswith("Bar Chart"):
                col = selected_chart.split("Bar Chart of ")[1]
                top_values = df[col].value_counts().head(15).reset_index()
                top_values.columns = [col, "Count"]
                top_values["Percentage"] = (top_values["Count"] / top_values["Count"].sum() * 100).round(1)
                colors = ["red" if i == 0 else "gray" for i in range(len(top_values))]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=top_values[col], y=top_values["Count"],
                    marker_color=colors,
                    text=top_values["Percentage"].astype(str) + "%",
                    textposition="auto",
                    name="Count"
                ))
                st.plotly_chart(fig)

            elif selected_chart.startswith("Histogram"):
                col = selected_chart.split("Histogram of ")[1]
                data = df[col].dropna()
                counts, bins = np.histogram(data, bins="auto")
                max_idx = counts.argmax()
                min_idx = counts.argmin()
                fig = go.Figure()
                for i in range(len(counts)):
                    color = "red" if i == max_idx else ("green" if i == min_idx else "gray")
                    label = f"{counts[i]}" if i == max_idx or i == min_idx else ""
                    fig.add_trace(go.Bar(
                        x=[f"{round(bins[i],1)}–{round(bins[i+1],1)}"],
                        y=[counts[i]],
                        marker_color=color,
                        text=[label],
                        textposition="outside"
                    ))
                fig.update_layout(title=f"Distribution of {col}", xaxis_title=col, yaxis_title="Count")
                st.plotly_chart(fig)

            elif selected_chart == "Scatter Plot (Pick Two)":
                col1 = st.selectbox("X-axis", num_cols, key="scatter_x")
                col2 = st.selectbox("Y-axis", [col for col in num_cols if col != col1], key="scatter_y")
                scatter = px.scatter(df, x=col1, y=col2, trendline="ols",
                                     title=f"{col2} vs {col1} with Trendline")
                st.plotly_chart(scatter)

            elif selected_chart == "Faceted Chart by Category":
                metric = st.selectbox("Metric to Compare", num_cols, key="facet_metric")
                category = st.selectbox("Group by", cat_cols, key="facet_cat")
                if df[category].nunique() < 15:
                    box = px.box(df, x=category, y=metric, points="all",
                                 title=f"{metric} by {category}")
                    st.plotly_chart(box)
                else:
                    st.info(f"Too many unique values in '{category}' for meaningful comparison.")

        data_summary = f"Columns: {', '.join(df.columns)}\n\nDescriptive Statistics:\n{df.describe().to_string()}"
except Exception as e:
    st.error(f"Error reading file: {e}")

# --- Session State ---
if "insight_text" not in st.session_state:
    st.session_state.insight_text = ""
if "show_followup_ui" not in st.session_state:
    st.session_state.show_followup_ui = False

# --- Generate Insight ---
if st.button("🤖 Generate AI-Powered Business Insights"):
    if not user_input:
        st.warning("Please enter a business problem.")
    else:
        with st.spinner("Generating insight..."):
            try:
                dataset_info = f"A supporting dataset was also provided:\n\n{data_summary}" if data_summary else ""

                prompt = (
                    f"You're a professional Business Analyst AI assistant.\n"
                    f"A stakeholder has shared the following business context:\n"
                    f"{user_input}\n\n"
                    f"{dataset_info}\n\n"
                    "Based on this, generate a structured business insight report including:\n"
                    "1. Business Requirements\n"
                    "2. Stakeholder Requirements\n"
                    "3. Key Observations from Data (if any)\n"
                    "4. Recommended Tasks\n"
                    "5. Risks or Challenges\n"
                    "6. Suggested Analysis Techniques or Next Steps\n"
                    "7. Solution Evaluation:\n"
                    "   - Briefly compare the current state after solution implementation with the previous (pre-solution) state.\n"
                    "   - Assess whether the solution met its intended goals.\n"
                    "   - Identify any internal limitations of the solution itself.\n"
                    "   - Highlight organizational or external constraints affecting performance.\n"
                    "   - Recommend actions to improve or optimize value delivery."
                )

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )

                insight_text = response.choices[0].message.content.strip()
                st.session_state.insight_text = insight_text
                st.session_state.show_followup_ui = True

            except Exception as e:
                st.error(f"❌ Error generating insight: {e}")

# --- Show Insight ---
if st.session_state.insight_text:
    with st.expander("📄 View Full Business Insight Report", expanded=True):
        st.markdown(st.session_state.insight_text)

# --- Follow-up ---
if st.session_state.show_followup_ui:
    st.markdown("### 💬 Ask about this insight")
    followup_input = st.text_input("Have a question about the insight?", key="followup_q")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("❓ Suggested: Explain Risks Section"):
            followup_input = "Can you explain the risks or challenges section?"
    with col2:
        if st.button("❓ Suggested: Expand on Recommended Tasks"):
            followup_input = "Can you elaborate on the recommended tasks section?"
    with col3:
        if st.button("✅ Solution Evaluation Insights"):
            followup_input = (
                "Please critically evaluate the Solution Evaluation section in the insight report. "
                "Compare the current solution to the previous state before implementation, "
                "and analyze whether the solution achieved its original goals using any relevant data provided. "
                "If the section is vague, incomplete, or generic, suggest specific ways to improve its depth and relevance."
            )

    if followup_input:
        with st.spinner("Retrieving clarification..."):
            clarification_prompt = f"""
The user has a follow-up question:
\"{followup_input}\"

Here is the full business insight report:
\"\"\"{st.session_state.insight_text}\"\"\"

Please provide a clear, helpful answer based on this insight.
"""
            followup_resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": clarification_prompt}],
                temperature=0.5,
            )

            with st.container():
                st.markdown("#### 🔍 Clarified Response:")
                st.markdown(followup_resp.choices[0].message.content)

# --- Footer ---
st.markdown("---")
st.markdown("<small>💡 Built with Python, Streamlit, OpenAI</small>", unsafe_allow_html=True)
