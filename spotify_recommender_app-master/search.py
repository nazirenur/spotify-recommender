import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit as st
import numpy as np

cid = "826b585b0fe847129e75f7b569f8f904"
secret ="3cdc8ea35c7548cc9e0f9026fb2ef05d"

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def search_pic(song_name, artist_name):
	find = song_name + " " + artist_name
	song = sp.search(find, limit=5)
	if len(song["tracks"]["items"]) != 0:
		album_pic = song["tracks"]["items"][0]["album"]["images"][1]["url"]
		demo_song = song["tracks"]["items"][0]["preview_url"]
		return album_pic, demo_song
	else:
		return -1

def aud_feat(song_name, artist_name):
	find = song_name + " " + artist_name
	song = sp.search(find, limit=5)

	danceability = sp.audio_features(song["tracks"]["items"][0]["id"])[0]["danceability"]
	energy = sp.audio_features(song["tracks"]["items"][0]["id"])[0]["energy"]
	loudness = sp.audio_features(song["tracks"]["items"][0]["id"])[0]["loudness"]
	speechiness = sp.audio_features(song["tracks"]["items"][0]["id"])[0]["speechiness"]
	acousticness = sp.audio_features(song["tracks"]["items"][0]["id"])[0]["acousticness"]
	instrumentalness = sp.audio_features(song["tracks"]["items"][0]["id"])[0]["instrumentalness"]
	liveness = sp.audio_features(song["tracks"]["items"][0]["id"])[0]["liveness"]
	valence = sp.audio_features(song["tracks"]["items"][0]["id"])[0]["valence"]
	tempo = sp.audio_features(song["tracks"]["items"][0]["id"])[0]["tempo"]

	api_song = pd.DataFrame(data=[danceability, energy, loudness,
	                                speechiness, acousticness, instrumentalness,
	                                liveness, valence, tempo],
	                        index=["danceability", "energy", "loudness", "speechiness", "acousticness",
	                               "instrumentalness", "liveness", "valence", "tempo"],
	                        columns=["x"]).T
	return api_song

def info(song_name, artist_name):
	find = song_name + " " + artist_name
	song = sp.search(find, limit=5)
	art_info = song["tracks"]["items"][0]["artists"][0]["name"]
	song_info = song["tracks"]["items"][0]["name"]
	song_spot = song["tracks"]["items"][0]["external_urls"]["spotify"]
	art_spot = song["tracks"]["items"][0]["album"]["artists"][0]["external_urls"]["spotify"]
	return art_info, song_info, song_spot, art_spot
