# 🎬 Emotion-Based Movie Recommender (Streamlit App)

This web app recommends movies or TV shows based on your **mood (emotion)** and **genre preferences**, using **PCA/LDA**, **TF-IDF**, and **cosine similarity**.

## 🚀 Features
- Emotion → Genre mapping (e.g. Happy → Comedy, Musical)
- Movie search by title, actor, or director
- PCA / LDA-based feature projections
- Cosine-similarity recommendations
- Genre-consistency evaluation
- "Surprise Me" button 🎲

## 🧠 Tech Stack
- Python, Streamlit
- Scikit-learn (TF-IDF, PCA, LDA)
- Pandas, NumPy, Matplotlib

## 🛠️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
