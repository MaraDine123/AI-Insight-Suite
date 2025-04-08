import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_lottie import st_lottie
from transformers import pipeline
import requests
import numpy as np
import traceback
import logging
import openai
from openai.embeddings_utils import get_embedding
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Page setup
st.set_page_config(page_title="💬 Feedback Analyzer", layout="wide")
openai.api_key = st.secrets["openai_key"]

# ✅ Load Lottie animation
def load_lottie_url(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_data(show_spinner=False)
def embed_feedback_list_hashable(texts):
    unique_key = hash(tuple(texts))
    return [get_embedding(t, engine="text-embedding-3-small") for t in texts]

sentiment_analyzer = load_sentiment_model()
lottie_feedback = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_1pxqjqps.json")

# ─────────── UI HEADER ─────────── #
with st.container():
    col1, col2 = st.columns([1, 2])
    with col1:
        if lottie_feedback:
            st_lottie(lottie_feedback, height=200)
    with col2:
        st.markdown("""
        <h1 style='color:#4CAF50;'>AI-Powered Customer Feedback Analyzer</h1>
        <p style='color:#555; font-size:18px;'>Analyze sentiment, tag intent, and ask questions from real feedback — lightning fast.</p>
        """, unsafe_allow_html=True)

# ─────────── FILE UPLOAD ─────────── #
uploaded_file = st.file_uploader("📁 Upload a feedback file (CSV or JSON)", type=["csv", "json"], key="feedback_uploader")

if uploaded_file:
    st.success(f"✅ Uploaded: `{uploaded_file.name}`")
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_json(uploaded_file)
        text_columns = [col for col in df.columns if df[col].dtype == 'object']
        text_col = st.selectbox("📝 Select feedback text column", text_columns)

        if text_col:
            # ─────────── Sentiment ─────────── #
            with st.spinner("🔍 Analyzing sentiment..."):
                df["sentiment_result"] = df[text_col].astype(str).apply(lambda x: sentiment_analyzer(x)[0]['label'])

            st.markdown("### 📊 Sentiment Breakdown")
            sentiment_counts = df["sentiment_result"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]
            fig = px.pie(sentiment_counts, values="Count", names="Sentiment",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

            # ─────────── Intent Tagging (Batch) ─────────── #
            st.markdown("### 🏷️ Feedback Intent Tagging")
            with st.spinner("🔎 Tagging feedback intent (batch)..."):
                try:
                    sample_rows = df[text_col].dropna().astype(str).tolist()[:30]
                    formatted = "\n".join([f"{i+1}. {line}" for i, line in enumerate(sample_rows)])
                    prompt = f"""Classify each feedback below as Complaint, Suggestion, or Praise.\n\n{formatted}\n\nRespond as:\n1. Complaint\n2. Praise\n..."""

                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are classifying feedback."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0,
                        max_tokens=300
                    )

                    result_text = response['choices'][0]['message']['content']
                    tags = [line.split(". ", 1)[-1] for line in result_text.strip().split("\n")]
                    df["intent_tag"] = "N/A"
                    df.loc[:len(tags)-1, "intent_tag"] = tags
                    st.dataframe(df[[text_col, "sentiment_result", "intent_tag"]].head(30), use_container_width=True)

                except Exception as e:
                    st.error("❌ Tagging failed.")
                    st.code(str(e))

            # ─────────── Executive Summary ─────────── #
            st.subheader("📊 Executive Summary")
            positive = df["sentiment_result"].str.lower().str.count("positive").sum()
            total = len(df)
            positive_pct = round((positive / total) * 100, 1) if total > 0 else 0
            feedback_text = " ".join(df[text_col].dropna().astype(str).tolist())[:3000]

            if total >= 5:
                with st.spinner("🧠 Writing executive summary..."):
                    try:
                        summary_prompt = f"""
Summarize the following feedback.\n\nFeedback:\n{feedback_text}\n\nInclude:\n📊 Sentiment: {positive_pct}% Positive  \n🧠 Key customer insight  \n💡 Recommendation
"""
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You summarize customer feedback for business analysts."},
                                {"role": "user", "content": summary_prompt}
                            ],
                            temperature=0.4,
                            max_tokens=250
                        )
                        st.markdown(response['choices'][0]['message']['content'])
                    except Exception as e:
                        st.error("❌ Failed to generate executive summary.")
                        st.code(str(e))

            # ─────────── Chatbot Q&A (FAST) ─────────── #
            st.subheader("🤖 Ask the AI About Customer Feedback")
            user_question = st.text_input("💬 Ask anything about the feedback")

            if user_question:
                with st.spinner("🤖 Thinking..."):
                    try:
                        feedback_texts = df[text_col].astype(str).tolist()
                        with st.spinner("Embedding feedback... please wait a moment."):
                            embeddings = embed_feedback_list_hashable(feedback_texts)

                        question_embed = get_embedding(user_question, engine="text-embedding-3-small")
                        similarities = cosine_similarity([question_embed], embeddings)[0]
                        top_indexes = similarities.argsort()[-3:][::-1]
                        top_texts = [feedback_texts[i][:250] for i in top_indexes]  # 🚀 Shortened for speed
                        context = "\n".join(top_texts)

                        chat_prompt = f"Based on this feedback:\n{context}\n\nAnswer:\n{user_question}"

                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You answer business questions from customer feedback."},
                                {"role": "user", "content": chat_prompt}
                            ],
                            temperature=0.5,
                            max_tokens=300
                        )
                        st.markdown(response['choices'][0]['message']['content'])

                    except Exception as e:
                        st.error("❌ Chatbot failed.")
                        st.code(str(e))

    except Exception as e:
        st.error("❌ Failed to process file.")
        st.code(traceback.format_exc())
else:
    st.warning("👆 Upload a feedback file to get started.")
