import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# ---------- 資料庫連線設定 ----------
def get_engine():
    return create_engine("mysql+mysqlconnector://root:@localhost/music_db")

# ---------- 歌曲搜尋 ----------
def search_songs(keyword, genre, year_range):
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql("""
            SELECT 
                s.title, 
                s.artist, 
                GROUP_CONCAT(g.name ORDER BY g.name SEPARATOR ', ') AS genres,
                YEAR(s.release_date) AS year,
                s.emotion
            FROM Songs s
            JOIN Song_Genres sg ON s.song_id = sg.song_id
            JOIN Genres g ON sg.genre_id = g.genre_id
            WHERE (s.title LIKE %(kw)s OR s.artist LIKE %(kw)s)
              AND (%(genre)s = '全部' OR g.name = %(genre)s)
              AND (s.release_date IS NOT NULL AND YEAR(s.release_date) BETWEEN %(min_year)s AND %(max_year)s)
            GROUP BY s.song_id, s.title, s.artist, s.release_date, s.emotion
            LIMIT 50
        """, conn, params={
            "kw": f"%{keyword}%",
            "genre": genre,
            "min_year": year_range[0],
            "max_year": year_range[1]
        })

    if not df.empty:
        df["YouTube"] = df.apply(
            lambda row: f"<a href='https://www.youtube.com/results?search_query={'+'.join(row['title'].split())}+{'+'.join(row['artist'].split())}' target='_blank'>\ud83d\udd17</a>",
            axis=1
        )
    return df

# ---------- 分群推薦 ----------
def get_cluster_recommendations():
    engine = get_engine()
    with engine.connect() as conn:
        df_all = pd.read_sql("""
            SELECT song_id, title, artist, energy, danceability, positiveness, speechiness,
                   liveness, acousticness, instrumentalness
            FROM Songs
            WHERE release_date IS NOT NULL
        """, conn)

    features = ["energy", "danceability", "positiveness", "speechiness",
                "liveness", "acousticness", "instrumentalness"]

    df_all = df_all.dropna(subset=features)
    scaler = StandardScaler()
    X = scaler.fit_transform(df_all[features])
    df_all["cluster"] = KMeans(n_clusters=10, random_state=42).fit_predict(X)

    sampled = df_all.groupby("cluster").apply(lambda g: g.sample(1, random_state=42)).reset_index(drop=True)
    sampled["YouTube"] = sampled.apply(
        lambda row: f"https://www.youtube.com/results?search_query={'+'.join(row['title'].split())}+{'+'.join(row['artist'].split())}",
        axis=1
    )
    return sampled

# ---------- 人格推論 ----------
def infer_personality(liked_ids):
    engine = get_engine()
    id_list = ",".join(str(i) for i in liked_ids)

    with engine.connect() as conn:
        avg_row = pd.read_sql(f"""
            SELECT AVG(energy) AS energy, AVG(danceability) AS danceability,
                   AVG(positiveness) AS positiveness, AVG(speechiness) AS speechiness,
                   AVG(liveness) AS liveness, AVG(acousticness) AS acousticness,
                   AVG(instrumentalness) AS instrumentalness
            FROM Songs
            WHERE song_id IN ({id_list})
        """, conn).iloc[0]

        df_types = pd.read_sql("SELECT * FROM Personality_Types", conn)

    def classify(val): return "low" if val < 40 else "mid" if val < 70 else "high"
    user_levels = {k: classify(avg_row[k]*100) for k in avg_row.index}

    def score(row):
        return sum(row[f"{f}_level"] == user_levels[f] for f in user_levels)

    df_types["match_score"] = df_types.apply(score, axis=1)
    df_types["match_percent"] = df_types["match_score"] / df_types["match_score"].sum()

    best = df_types.sort_values("match_score", ascending=False).iloc[0]
    return best["personality_type"], best["description"], df_types

# ---------- 正規化 ----------
def normalize_feature_series(avg_row, stats):
    result = {}
    for feature in avg_row.index:
        val = avg_row[feature]
        max_val = stats.get(f"{feature}_max", 1)
        min_val = stats.get(f"{feature}_min", 0)
        if pd.isna(val) or pd.isna(max_val) or pd.isna(min_val):
            result[feature] = 0.0
        elif max_val == min_val:
            result[feature] = 0.0
        else:
            result[feature] = (val - min_val) / (max_val - min_val)
    return pd.Series(result)

# ---------- 雷達圖 ----------
def plot_radar_chart_plotly(vector, title="你的音樂特徵雷達圖"):
    feature_map = {
        "energy": "活力", "danceability": "舞動性", "positiveness": "正向情緒",
        "speechiness": "語音成分", "liveness": "現場感", "acousticness": "原聲程度",
        "instrumentalness": "器樂性"
    }
    feature_keys = list(feature_map.keys())
    vector = vector[feature_keys].fillna(0).astype(float)

    labels = [feature_map[k] for k in feature_keys]
    values = vector.values.tolist()
    labels += labels[:1]
    values += values[:1]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=labels, fill='toself',
        line=dict(color='rgba(0,123,255,0.8)', width=3),
        fillcolor='rgba(0,123,255,0.2)'
    ))
    fig.update_layout(
        polar=dict(bgcolor='white', radialaxis=dict(visible=True, range=[0, 1])),
        paper_bgcolor='white', plot_bgcolor='white',
        showlegend=False, title=title
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------- 長條圖：人格相似度 ----------
def plot_personality_match_bar(df_types):
    df_plot = df_types[["personality_type", "match_percent"]].copy()
    df_plot = df_plot.sort_values("match_percent", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_plot["match_percent"] * 100,
        y=df_plot["personality_type"],
        orientation="h",
        text=[f"{p:.1%}" for p in df_plot["match_percent"]],
        textposition="auto",
        marker_color='rgba(0,123,255,0.7)'
    ))
    fig.update_layout(
        title="人格型態的相似度分析",
        xaxis_title="相似度 (%)",
        yaxis_title="人格型態",
        plot_bgcolor='white', paper_bgcolor='white', height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------- Streamlit 主程式 ----------
def main():
    st.set_page_config(page_title="音樂推薦與人格預測", layout="wide")

    st.sidebar.title("功能選單")
    page = st.sidebar.radio("請選擇功能", ["🔍 搜尋歌曲", "🎵 人格推薦"])

    if page == "🔍 搜尋歌曲":
        st.header("🎵 音樂查詢系統")
        keyword = st.text_input("關鍵字（歌名或歌手）")

        with get_engine().connect() as conn:
            genre_options = ["全部"] + pd.read_sql("SELECT name FROM Genres ORDER BY name", conn)['name'].dropna().tolist()

        genre = st.selectbox("音樂類型", genre_options)
        year_range = st.slider("年代範圍", 1950, 2025, (2000, 2023))

        if st.button("搜尋"):
            if keyword.strip() == "":
                st.warning("請輸入關鍵字")
            else:
                results = search_songs(keyword, genre, year_range)
                if results.empty:
                    st.info("找不到符合條件的歌曲。")
                else:
                    st.markdown(results.to_html(escape=False, index=False), unsafe_allow_html=True)

    elif page == "🎵 人格推薦":
        st.header("🎧 勾選喜好歌曲，預測你的音樂人格")

        df = get_cluster_recommendations()
        selected = []

        for _, row in df.iterrows():
            col1, col2 = st.columns([0.05, 0.95])
            checked = col1.checkbox("", key=row["song_id"])
            col2.markdown(f"**{row['title']} - {row['artist']}** [🔗]({row['YouTube']})", unsafe_allow_html=True)
            if checked:
                selected.append(row["song_id"])

        if st.button("送出喜好"):
            if not selected:
                st.warning("請至少勾選一首你喜歡的歌曲")
            else:
                personality, desc, df_types = infer_personality(selected)
                st.success(f"你可能的人格類型是：**{personality}**")
                st.write(f"描述：{desc}")

                engine = get_engine()
                with engine.connect() as conn:
                    user_avg = pd.read_sql(f"""
                        SELECT AVG(energy) AS energy, AVG(danceability) AS danceability,
                               AVG(positiveness) AS positiveness, AVG(speechiness) AS speechiness,
                               AVG(liveness) AS liveness, AVG(acousticness) AS acousticness,
                               AVG(instrumentalness) AS instrumentalness
                        FROM Songs
                        WHERE song_id IN ({','.join(str(i) for i in selected)})
                    """, conn).iloc[0]

                df_recommended = get_cluster_recommendations()
                features = ["energy", "danceability", "positiveness", "speechiness",
                            "liveness", "acousticness", "instrumentalness"]
                stats = {f"{f}_max": df_recommended[f].max() for f in features}
                stats.update({f"{f}_min": df_recommended[f].min() for f in features})

                user_avg = normalize_feature_series(user_avg, stats)
                plot_radar_chart_plotly(user_avg, title="🎧 你的音樂特徵雷達圖")
                plot_personality_match_bar(df_types)

if __name__ == "__main__":
    main()

