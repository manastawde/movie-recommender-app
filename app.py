import streamlit as st
import pandas as pd
import numpy as np
import os, re, random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

st.set_page_config(page_title="Final Movie Recommender", layout="wide")

CSV_PATH = "IMDb_All_Genres_Movies_cleaned.csv"
RANDOM_STATE = 42

# ---------- Utilities ----------
def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at: {path}")
    return pd.read_csv(path)

def normalize_string(s):
    return re.sub(r'\s+', ' ', str(s).strip())

def split_genres_field(s):
    if pd.isna(s) or not str(s).strip():
        return []
    return [p.strip().title() for p in re.split(r'[,/;|]', str(s)) if p.strip()]

def clean_df(df):
    df = df.copy()
    for c in ["Movie_Title","Actors","Director","main_genre","side_genre","Rating","Runtime(Mins)","Year"]:
        if c not in df.columns:
            df[c] = ""
    df["Movie_Title"] = df["Movie_Title"].astype(str).apply(normalize_string)
    df["Actors"] = df["Actors"].astype(str).fillna("")
    df["Director"] = df["Director"].astype(str).fillna("")
    df["main_genre"] = df["main_genre"].astype(str).fillna("")
    df["side_genre"] = df["side_genre"].astype(str).fillna("")

    df["main_tokens"] = df["main_genre"].apply(split_genres_field)
    df["side_tokens"] = df["side_genre"].apply(split_genres_field)
    df["all_genres"] = df.apply(lambda r: sorted(set(r["main_tokens"] + r["side_tokens"])), axis=1)
    df["primary_genre"] = df["all_genres"].apply(lambda arr: arr[0] if arr else "Unknown")

    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(0.0)
    df["Runtime(Mins)"] = pd.to_numeric(df["Runtime(Mins)"], errors="coerce").fillna(0.0)
    def safe_year(y):
        try:
            return int(float(y))
        except Exception:
            return 0
    df["Year"] = df["Year"].apply(safe_year)

    df["combined_text"] = df["Movie_Title"] + " | " + df["Actors"] + " | " + df["Director"] + " | " + df["main_genre"] + " | " + df["side_genre"]
    df["title_key"] = df["Movie_Title"].str.lower().str.replace(r'[^a-z0-9 ]','',regex=True).str.strip()
    df = df.sort_values(["Rating","Runtime(Mins)"], ascending=[False, True])
    df = df.drop_duplicates(subset=["title_key"], keep="first").reset_index(drop=True)
    return df

# ---------- Feature building ----------
@st.cache_data
def build_tfidf_svd(corpus, max_features=5000):
    n_components = min(50, max(5, len(corpus)//4))
    vec = TfidfVectorizer(stop_words="english", max_features=max_features)
    X = vec.fit_transform(corpus)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    X_red = svd.fit_transform(X)
    return vec, svd, X_red

def build_final_projection(features, method="PCA", n_components=10, labels=None):
    if method == "PCA":
        model = PCA(n_components=min(n_components, features.shape[1]), random_state=RANDOM_STATE)
        return model.fit_transform(features), model
    else:
        if labels is None or len(np.unique(labels)) < 2:
            model = PCA(n_components=min(n_components, features.shape[1]), random_state=RANDOM_STATE)
            return model.fit_transform(features), model
        ncomp = min(n_components, max(1, len(np.unique(labels))-1), features.shape[1])
        try:
            lda = LinearDiscriminantAnalysis(n_components=ncomp)
            return lda.fit_transform(features, labels), lda
        except Exception:
            model = PCA(n_components=min(n_components, features.shape[1]), random_state=RANDOM_STATE)
            return model.fit_transform(features), model

def recommend_by_index(index, final_space, df_filtered, top_n=8):
    sims = cosine_similarity(final_space[index:index+1], final_space).flatten()
    order = np.argsort(-sims)
    recs = []
    seen = set()
    for i in order:
        if i == index:
            continue
        title = df_filtered.iloc[i]["Movie_Title"]
        if title in seen:
            continue
        seen.add(title)
        recs.append((i, title, float(sims[i])))
        if len(recs) >= top_n:
            break
    return recs

def evaluate_genre_consistency(final_space, df_filtered, top_k=5):
    sims = cosine_similarity(final_space)
    n = final_space.shape[0]
    match_ratios, sim_scores = [], []
    for i in range(n):
        order = np.argsort(-sims[i])
        top = [j for j in order if j != i][:top_k]
        if not top:
            continue
        target = df_filtered.iloc[i]["primary_genre"]
        same = sum(1 for j in top if df_filtered.iloc[j]["primary_genre"] == target)
        match_ratios.append(same / top_k)
        sim_scores.append(np.mean([sims[i,j] for j in top]))
    return float(np.mean(match_ratios)) if match_ratios else 0.0, float(np.mean(sim_scores)) if sim_scores else 0.0

# ---------- New: emotion -> genre mapping ----------
EMOTION_TO_GENRES = {
    "Happy": ["Comedy","Family","Musical"],
    "Sad": ["Drama","Romance"],
    "Angry": ["Action","Crime","Thriller"],
    "Calm": ["Documentary","Drama","Romance"],
    "Excited": ["Action","Adventure"],
    "Romantic": ["Romance","Drama"],
    "Scared": ["Horror","Thriller"],
    "Nostalgic": ["Drama","History"],
    "Curious": ["Mystery","Sci-Fi","Documentary"],
    "Bored": ["Comedy","Adventure","Action"]
}

# ---------- Small palette for genre badges ----------
PALETTE = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f",
    "#bcbd22","#17becf","#393b79","#637939","#8c6d31","#843c39","#7b4173","#3182bd"
]
def genre_color_map(genres):
    unique = sorted(list(genres))
    cmap = {}
    for i,g in enumerate(unique):
        cmap[g] = PALETTE[i % len(PALETTE)]
    return cmap

def badge_html(text, color):
    return f'<span style="background:{color};color:#fff;padding:3px 8px;border-radius:10px;font-size:12px;">{text}</span>'

# ---------- App ----------
st.title("🎬 Final Movie Recommender (Clean & Simple)")

# Load + clean
try:
    raw = load_csv(CSV_PATH)
except FileNotFoundError as e:
    st.error(str(e)); st.stop()
df = clean_df(raw)

# ================= Sidebar =================
with st.sidebar:
    st.header("Controls")
    st.markdown("### 1) Emotion (optional)")
    emotion = st.selectbox("How are you feeling?", options=[""] + list(EMOTION_TO_GENRES.keys()))
    st.markdown("---")
    st.markdown("### 2) Search")
    search_q = st.text_input("Search by title / actor / director (press Enter)")
    st.markdown("---")
    st.markdown("### 3) Projection & filters")
    tfidf_max = st.number_input("TF-IDF max features", min_value=200, max_value=20000, value=5000, step=100)
    final_method = st.radio("Final projection", ("PCA", "LDA"))
    final_comp = st.number_input("Final components", min_value=2, max_value=50, value=10, step=1)
    min_rating = st.slider("Min rating", 0.0, 10.0, 0.0, step=0.1)
    year_min, year_max = st.slider("Year range", int(df["Year"].min()), int(df["Year"].max()), (int(df["Year"].min()), int(df["Year"].max())))
    st.markdown("---")
    st.markdown("### 4) Recommendation options")
    chosen_genre = st.selectbox("Filter by genre (optional)", options=[""] + sorted({g for arr in df["all_genres"] for g in arr}))
    rec_count = st.number_input("Number of recommendations", min_value=1, max_value=50, value=8, step=1)
    st.markdown("---")
    st.markdown("### 5) Utilities")
    if st.button("Surprise me 🎲 (high-rated)"):
        st.session_state.get("surprise", True)
    else:
        # clear flag so it only triggers when pressed
        if "surprise" in st.session_state:
            st.session_state.pop("surprise", None)

# ================= Apply search filter / emotion =================
# Apply base filters
mask = (df["Rating"] >= min_rating) & df["Year"].between(year_min, year_max)

# If emotion selected, filter by mapped genres (but allow override with chosen_genre)
if emotion:
    mapped = EMOTION_TO_GENRES.get(emotion, [])
    if mapped:
        # keep movies that have at least one mapped genre in their all_genres
        mask &= df["all_genres"].apply(lambda arr: any(m in arr for m in mapped))

# If user selected explicit genre in dropdown, further filter
if chosen_genre:
    mask &= df["all_genres"].apply(lambda arr: chosen_genre in arr)

# Apply search query: find any row where title/actors/director contains tokens
if search_q and search_q.strip():
    q = search_q.strip().lower()
    mask &= df.apply(lambda r: (q in str(r["Movie_Title"]).lower()) or (q in str(r["Actors"]).lower()) or (q in str(r["Director"]).lower()), axis=1)

filtered = df[mask].reset_index(drop=True)
st.write(f"Filtered movies: {len(filtered)}")
if len(filtered) == 0:
    st.warning("No movies after filtering. Adjust filters or clear search/emotion.")
    st.stop()

# If Surprise Me triggered, pick a top-rated random movie from filtered (and set as sel)
surprise_mode = st.session_state.get("surprise", False)
surprise_choice = None
if surprise_mode:
    # pick among top 25% ratings in filtered
    cutoff = np.quantile(filtered["Rating"].values, 0.75) if len(filtered) > 3 else filtered["Rating"].max()
    candidates = filtered[filtered["Rating"] >= cutoff]
    if len(candidates) == 0:
        candidates = filtered
    surprise_choice = candidates.sample(1, random_state=RANDOM_STATE).iloc[0]["Movie_Title"]
    # clear flag so repeated reruns don't auto-repeat unless button pressed again
    if "surprise" in st.session_state:
        st.session_state.pop("surprise", None)
    st.success(f"Surprise pick: **{surprise_choice}**")

# Build TF-IDF + SVD
with st.spinner("Building embeddings..."):
    tfidf_vec, svd, _ = build_tfidf_svd(df["combined_text"].tolist(), max_features=tfidf_max)
    Xf = tfidf_vec.transform(filtered["combined_text"].tolist())
    text_vectors = svd.transform(Xf)

# Combine numeric
num = np.vstack([filtered["Rating"].to_numpy(), filtered["Runtime(Mins)"].to_numpy(), filtered["Year"].to_numpy()]).T
num_scaled = StandardScaler().fit_transform(num)
features = np.hstack([text_vectors, num_scaled])

# Final projection
labels = filtered["primary_genre"].values if final_method == "LDA" else None
with st.spinner("Applying final projection..."):
    final_space, final_model = build_final_projection(features, final_method, final_comp, labels)

# Visualization (matplotlib)
st.subheader("2D projection (first 2 components)")
if final_space.shape[1] >= 2:
    fig, ax = plt.subplots(figsize=(7,5))
    genres = filtered["primary_genre"].astype(str).values
    unique_genres = sorted(list(set(genres)))
    cmap = genre_color_map(unique_genres)
    colors = [cmap.get(g, "#777777") for g in genres]
    scatter = ax.scatter(final_space[:,0], final_space[:,1], c=colors, s=24, edgecolor='k', linewidth=0.2)
    ax.set_xlabel("Comp 1"); ax.set_ylabel("Comp 2")
    # legend (genre color badges)
    handles = []
    for g in unique_genres[:16]:  # limit legend length visually
        handles.append(plt.Line2D([0],[0], marker='o', color='w', label=g,
                                  markerfacecolor=cmap[g], markersize=8))
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left', title="Primary genre")
    st.pyplot(fig)
else:
    st.info("Increase final components to at least 2 to view scatter.")

# ================= Recommendations UI =================
st.markdown("---")
st.header("Recommendations")

# helper to render title with genre badge
genre_colors = genre_color_map({g for arr in filtered["all_genres"] for g in arr})

def show_recs(recs, prefix=""):
    for rank,(i,title,score) in enumerate(recs, start=1):
        r = filtered.iloc[i]
        badge = badge_html(r['primary_genre'], genre_colors.get(r['primary_genre'], "#444444"))
        # show badge and title (unsafe HTML allowed)
        st.markdown(f"{rank}. {badge}  <b>{title}</b> — ⭐ {r['Rating']} | {r['Year']} — Sim: {score:.3f}", unsafe_allow_html=True)
        # small details
        st.write(f"Director: {r['Director']} | Runtime: {int(r['Runtime(Mins)'])} mins")
        # More like this
        if st.button(f"More like this → {title}", key=f"{prefix}_{i}"):
            more = recommend_by_index(i, final_space, filtered, top_n=rec_count)
            st.markdown(f"**More like {title}:**")
            for rr, (ii, t2, s2) in enumerate(more, start=1):
                rrw = filtered.iloc[ii]
                b2 = badge_html(rrw['primary_genre'], genre_colors.get(rrw['primary_genre'], "#444444"))
                st.markdown(f"- {rr}. {b2} <b>{t2}</b> — ⭐ {rrw['Rating']} | {rrw['Year']} — Sim: {s2:.3f}", unsafe_allow_html=True)
        st.markdown("---")

mode = st.radio("Query source", ("Existing movie", "Custom entry"), index=0)

if mode == "Existing movie":
    # if Surprise Me chosen, preselect it, else normal selectbox
    if surprise_choice:
        sel = surprise_choice
    else:
        sel = st.selectbox("Pick a movie", options=[""] + filtered["Movie_Title"].tolist())
    if sel:
        pos = filtered.index[filtered["Movie_Title"] == sel][0]
        recs = recommend_by_index(pos, final_space, filtered, top_n=rec_count)
        st.write(f"Top {len(recs)} similar to **{sel}**:")
        show_recs(recs, "sel")
else:
    st.write("Provide a custom movie entry:")
    with st.form("custom"):
        c_title = st.text_input("Title", value="My Movie")
        c_actors = st.text_input("Actors", value="")
        c_director = st.text_input("Director", value="")
        c_main = st.text_input("Main genre", value="")
        c_side = st.text_input("Side genre", value="")
        c_rating = st.number_input("Rating", 0.0, 10.0, 6.0, step=0.1)
        c_runtime = st.number_input("Runtime", 0, 600, 120, step=1)
        c_year = st.number_input("Year", 1800, 2100, 2020, step=1)
        sub = st.form_submit_button("Find")
    if sub:
        combined = f"{c_title} | {c_actors} | {c_director} | {c_main} | {c_side}"
        q_text = tfidf_vec.transform([combined])
        q_text_red = svd.transform(q_text)
        q_num = np.array([[c_rating, c_runtime, c_year]])
        q_num_scaled = StandardScaler().fit(num).transform(q_num)
        q_feat = np.hstack([q_text_red, q_num_scaled])
        try:
            q_proj = final_model.transform(q_feat)
        except Exception:
            q_proj = PCA(n_components=min(final_comp, q_feat.shape[1])).fit_transform(q_feat)
        sims = cosine_similarity(q_proj, final_space).flatten()
        idxs = np.argsort(-sims)[:rec_count]
        recs = [(int(i), filtered.iloc[int(i)]["Movie_Title"], float(sims[i])) for i in idxs]
        st.write(f"Top {len(recs)} recommendations for your custom entry:")
        show_recs(recs, "custom")

# ================= Accuracy Test =================
st.markdown("---")
st.header("Model accuracy")
if st.button("Run genre-consistency test (Top-5)"):
    with st.spinner("Evaluating..."):
        gacc, avg_sim = evaluate_genre_consistency(final_space, filtered, 5)
        st.write(f"🎯 Genre Consistency: {gacc*100:.2f}%")
        st.write(f"🔗 Avg Cosine Similarity: {avg_sim:.4f}")
