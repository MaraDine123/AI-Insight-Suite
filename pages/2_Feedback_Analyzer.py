import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from transformers import pipeline
from streamlit_lottie import st_lottie
from pinecone import Pinecone, ServerlessSpec
import uuid
import openai

# -------------------- CONFIG -------------------- #
st.set_page_config(page_title="💬 Feedback Analyzer", layout="wide")

# --------------------- CACHE --------------------- #
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_data(show_spinner=False)
def load_lottie_url(url: str):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

def main():
    try:
        openai.api_key = st.secrets["OPENAI_KEY"]
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

        if st.secrets["PINECONE_INDEX"] not in pc.list_indexes().names():
            pc.create_index(
                name=st.secrets["PINECONE_INDEX"],
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=st.secrets["PINECONE_ENVIRONMENT"]
                )
            )

        index = pc.Index(st.secrets["PINECONE_INDEX"])

    except KeyError as e:
        st.error(f"🚨 Missing key in Streamlit secrets: {e}")
        st.stop()

    sentiment_analyzer = load_sentiment_model()
    lottie_feedback = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_1pxqjqps.json")

    # --------------------- HEADER --------------------- #
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            if lottie_feedback:
                st_lottie(lottie_feedback, height=200)
        with col2:
            st.markdown("""
                <h1 style='color:#4CAF50;'>AI-Powered Customer Feedback Analyzer</h1>
                <p style='font-size:17px;'>Turn feedback into business insights using sentiment analysis, intent tagging, and smart AI recommendations.</p>
            """, unsafe_allow_html=True)

    # --------------------- FILE UPLOAD --------------------- #
    st.subheader("📅 Upload Your Feedback File")
    uploaded_file = st.file_uploader("Upload CSV or JSON file", type=["csv", "json"], key="uploader")

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_json(uploaded_file)

        text_columns = [col for col in df.columns if df[col].dtype == 'object']
        if not text_columns:
            st.error("❌ No text-based column found in file.")
            st.stop()

        text_col = text_columns[0]

        # --------------------- SENTIMENT --------------------- #
        df["sentiment_result"] = df[text_col].astype(str).apply(lambda x: sentiment_analyzer(x)[0]['label'].upper())

        st.subheader("📊 Sentiment Breakdown")
        sentiment_counts = df["sentiment_result"].value_counts().reset_index()
        sentiment_counts.columns = ["sentiment", "count"]

        if df["sentiment_result"].nunique() > 3:
            fig = px.bar(sentiment_counts, x="sentiment", y="count")
        else:
            fig = px.pie(sentiment_counts, names="sentiment", values="count")
        st.plotly_chart(fig, use_container_width=True)

        # --------------------- INTENT TAGGING --------------------- #
        st.subheader("🏷️ Feedback Intent Tagging")
        def classify_intent(text):
            text = text.lower()
            if any(w in text for w in ["not", "bad", "slow", "crash", "issue", "problem"]):
                return "Complaint"
            elif any(w in text for w in ["should", "wish", "add", "suggest"]):
                return "Suggestion"
            elif any(w in text for w in ["love", "great", "excellent", "like"]):
                return "Praise"
            return "Other"

        df["intent"] = df[text_col].apply(classify_intent)
        st.dataframe(df[[text_col, "sentiment_result", "intent"]], use_container_width=True)

        # --------------------- INSIGHT FROM OPENAI --------------------- #
        st.subheader("📡 Customer Feedback Insights")
        if st.button("Fetch the Analysis"):
            with st.spinner("🔎 Analyzing uploaded feedback with AI..."):
                feedback_list = df[text_col].dropna().astype(str).tolist()
                combined_feedback = "\n".join(feedback_list[:30])
                analysis_prompt = f"""
You're a senior customer experience analyst. The following customer feedbacks have been collected:
{combined_feedback}

Generate a structured insight report that includes:
1. Key issues raised by customers (with short reasoning).
2. Common suggestions or feature requests.
3. Any praise or positive sentiments.
4. Hypotheses on root causes of recurring issues.
5. Business impact (if observable).
6. Suggested prioritized actions.
7. Any product or experience gaps.
"""

                ai_response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.5,
                    max_tokens=1000
                )
                summary = ai_response.choices[0].message.content
                st.session_state["ai_feedback_summary"] = summary
                st.success("✅ Insight generated successfully!")

                def get_openai_embeddings(texts):
                    response = openai.embeddings.create(model="text-embedding-ada-002", input=texts)
                    return [e.embedding for e in response.data]

                with st.spinner("📥 Syncing feedback to Pinecone for Q&A..."):
                    vectors = get_openai_embeddings(feedback_list)
                    metadata = [{"text": f} for f in feedback_list]
                    ids = [str(uuid.uuid4()) for _ in feedback_list]
                    index.upsert(vectors=[(id, vec, meta) for id, vec, meta in zip(ids, vectors, metadata)])

        # --------------------- SHOW INSIGHT ALWAYS --------------------- #
        if "ai_feedback_summary" in st.session_state:
            st.markdown("### 🧠 Summary of Customer Feedback")
            st.markdown(st.session_state["ai_feedback_summary"])

            # --------------------- AI CHATBOT --------------------- #
            st.subheader("🤖 Ask the AI About Customer Feedback")
            query = st.text_input("Ask a question")

            if query:
                with st.spinner("🔍 Fetching relevant feedback..."):
                    def get_openai_embeddings(texts):
                        response = openai.embeddings.create(model="text-embedding-ada-002", input=texts)
                        return [e.embedding for e in response.data]

                    q_embed = get_openai_embeddings([query])
                    results = index.query(vector=q_embed[0], top_k=5, include_metadata=True)
                    context = "\n".join([match["metadata"]["text"] for match in results["matches"]])

                with st.spinner("💬 Generating response..."):
                    response = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": f"You're a customer feedback expert. Use this context to help answer user questions:\n{context}"},
                            {"role": "user", "content": query + " Please explain in a detailed manner with an example."}
                        ],
                        temperature=0.4,
                        max_tokens=700
                    )
                    st.markdown(response.choices[0].message.content)
    else:
        st.info("👉 Upload a feedback file (CSV or JSON) to begin.")

if __name__ == "__main__":
    main()
