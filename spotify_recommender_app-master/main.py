from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import search as src
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Spotify-Recommender App",
    page_icon="🎧",
    layout="wide"
)

if 'selected_match' not in st.session_state:
    st.session_state['selected_match'] = None
@st.cache_data
def get_data():
    df = pd.read_csv("data/last_data_spotify.csv")
    df = df.drop(["Unnamed: 0", "duration_ms"], axis=1)
    return df

df=get_data()
spo_jpg = Image.open("images/spotify2.jpg")
spo1_jpg = Image.open("images/spotify4.jpg")
spo6_jpg = Image.open("images/spotify6.jpg")
not_found_png = Image.open("images/notfound.png")

def scaled(DataFrame):
    sc = MinMaxScaler((0, 1))
    sc.fit(songs_list)
    DataFrame = sc.transform(DataFrame)
    return DataFrame
@st.cache_data
def df_split(data):
    features_list = ['danceability', 'energy', 'loudness', 'acousticness', "speechiness",
                     'instrumentalness', 'liveness', 'valence', 'tempo']
    songs_list = data[features_list]
    songs_list.index = list(data["track_id"])
    visual_list = data
    return songs_list, visual_list, features_list
songs_list, visual_list, features_list = df_split(df)

@st.cache_data
def cluster_main():
    songs_list_scaled = pd.DataFrame(scaled(songs_list), columns=features_list,index=songs_list.index)
    kmeans_1 = KMeans(n_clusters=7, random_state=1).fit(songs_list_scaled[['danceability', 'loudness', 'liveness', 'tempo', "speechiness"]])
    kmeans_2 = KMeans(n_clusters=7, random_state=1).fit(songs_list_scaled[['acousticness', 'valence']])
    songs_list_scaled["cluster"] = np.core.defchararray.add(kmeans_1.labels_.astype(str),kmeans_2.labels_.astype(str))
    return songs_list_scaled,kmeans_1,kmeans_2
songs_list_scaled,kmeans_1,kmeans_2= cluster_main()

def markdown_summary(col, art_info, song_info,song_spot, art_spot):
    return col.markdown(f"""
    Sanatçı Adı: {art_info}

    Şarkı Adı: {song_info}

    Şarkının Spotify Linki: {song_spot}

    Sanatçının Spotify Linki: {art_spot}
    """)

def input_df(DataFrame):
    api_song = pd.DataFrame(scaled(DataFrame[features_list]), columns=features_list, index=DataFrame.index)
    return api_song

def Item_Recommenddation(api_song, data, x):
    recommend_list = data[data["cluster"] == api_song["cluster"][0]]
    cosine_sim = cosine_similarity(recommend_list.drop(["cluster"], axis=1),
                                   api_song.drop(["cluster"], axis=1))
    recommend_list["cosine_sim"] = cosine_sim
    similary_list = recommend_list["cosine_sim"].sort_values(ascending=False).head(x)

    return similary_list

def main():
    col1, col2, col3 = st.columns(3)
    st.image(spo6_jpg,use_column_width="always", caption='S P O T I F Y')
    st.markdown("<h1 style='text-align: center; color: green;'>S p o t i f y  R e c o m m e n d e r</h1>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center;font-size:18px; color: gray ;'> 💖 F  o  r   y  o  u,  "
                "w  i  t  h    y  o  u  💖</h2>", unsafe_allow_html=True)

    first1,first2,first3 = st.columns(3, gap="medium")
    st.write(" ")
    first2.markdown("""Want to Find Songs You'll Love?""", unsafe_allow_html=True)
    song_name = first2.text_input("Song Name")
    artist_name = first2.text_input("Artist Name")
    check = first2.button("Search the Song..")

    if check and (len(song_name) >= 1 or len(artist_name) >= 1):
        if src.search_pic(song_name, artist_name) == -1:
            first2.markdown("""🤔""")
            first2.image(not_found_png, width=280)
        else:
            pic, demo = src.search_pic(song_name, artist_name)
            first1.markdown("Song You Liked:")
            first1.image(pic)
            first1.audio(demo)
            art_info, song_info, song_spot, art_spot = src.info(song_name, artist_name)
            markdown_summary(first1, art_info, song_info,song_spot, art_spot)

            input_song_df = src.aud_feat(song_name, artist_name)
            scaled_input_df = input_df(input_song_df)

            song_predict_1 = kmeans_1.predict(scaled_input_df[['danceability', 'loudness', 'liveness', 'tempo', "speechiness"]])
            song_predict_2 = kmeans_2.predict(scaled_input_df[['acousticness', 'valence']])
            scaled_input_df["cluster"] = np.core.defchararray.add(song_predict_1.astype(str),
                                                                 song_predict_2.astype(str))

            similary_list = Item_Recommenddation(scaled_input_df, songs_list_scaled, 10)
            son_dem = pd.DataFrame(similary_list)
            son_dem.reset_index(inplace=True)
            son_dem = son_dem.rename(columns={'index': 'track_id'})
            df['track_id'] = df['track_id'].astype('str')
            son_dem['track_id'] = son_dem['track_id'].astype('str')
            merged_df = pd.merge(df, son_dem, on='track_id', how='inner')
            rec_song = merged_df.iloc[0][2]
            rec_name = merged_df.iloc[0][1]
            rec_img, rec_preview = src.search_pic(rec_name, rec_song)
            first3.markdown("Song You Will Like..")
            first3.image(rec_img)
            first3.audio(rec_preview)
            art_info, song_info, song_spot, art_spot = src.info(rec_song, rec_name)
            markdown_summary(first3, art_info, song_info,  song_spot, art_spot)

            first2.image(spo1_jpg, use_column_width="always",width=175)
            st.markdown("<h1 style='text-align: center; color: gray'> You should add these to your list too.. </h1>", unsafe_allow_html=True)
            st.write("")
            list_cols = st.columns(5, gap="small")
            for i in range(5):
                rec_song = merged_df.iloc[i+1, 2]
                rec_name = merged_df.iloc[i+1, 1]
                rec_img, rec_preview = src.search_pic(rec_name, rec_song)
                list_cols[i].image(rec_img)
                list_cols[i].audio(rec_preview)
                art_info, song_info, song_spot, art_spot = src.info(rec_song, rec_name)
                markdown_summary(list_cols[i], art_info, song_info, song_spot, art_spot)

if __name__ == '__main__':
    main()



