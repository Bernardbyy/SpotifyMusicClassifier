import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import joblib
import pickle
import re
import time

#streamlit run app.py

# Load the fitted objects and model
le_song_title = joblib.load('label_encoder_song_title.pkl')
le_artist = joblib.load('label_encoder_artist.pkl')
scaler = joblib.load('standard_scaler.pkl')
xgb = joblib.load('xgb_Model.pkl')
known_song_titles = joblib.load('known_song_titles.pkl')
known_artists = joblib.load('known_artists.pkl')


# Spotify API setup
client_id = '6291dd76ffb744299366b81b9cddfe5c'
client_secret = '14e1d9b1399a4d8e8ea34e09717b4815'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Display the image from a URL
image_url = "https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_RGB_Green.png"

# Display the image
st.image(image_url, caption="Spotify Logo",use_column_width=True)

st.title("Spotify Track Predictor")

# Introduction and Instructions
st.markdown(
    """
    ## Welcome ! 🎶
    
    ### About This Project
    
    This web app is designed to predict the musical preferences of its creator based on a dataset of songs the creator has listened to and labeled. 
    It utilizes a machine learning model trained on various song features to make these predictions. 
    Curious to see how the model generalized with your songs? Give it a try!
    
    ### How to Use
    
    1. **Predict a Single Song**: To find out whether the model thinks the creator would like a specific song, simply input the Spotify URL of that song in the text box and hit "Enter"
      
    2. **Analyze a Playlist**: If you have a playlist you'd like the model to analyze, paste the Spotify Playlist URL in the designated text box. 
    The model will then predict whether the creator would like each song in the playlist.
    
    ### Example URLs for Testing
    Drag Link to Text Box / Copy URL
    - **Song (Like)**: [FEEL LIKE - Josh Fudge](https://open.spotify.com/track/6MFtTpEpk8Q2hZKKfid7SE?si=21c8f420a3a64abe)
    - **Song (Dislike)**: [Mia & Sebastian Theme - Justin Hurwitz](https://open.spotify.com/track/1Vk4yRsz0iBzDiZEoFMQyv?si=2bb3c40c02ef47a8)
    - **Example Playlist**: [Spotify Playlist](https://open.spotify.com/playlist/1PkLqSEP0rFaw6pw94DZbP?si=873990a54baf4e07)
    
    ### Let's Get Started!
    
    Scroll down to input your song or playlist URL and discover what the model predicts. Have fun!
    """
)

####################################################### SINGLE TRACK PREDICTION START ###########################################################
st.header("Predict Individual Tracks 🎶")

spotify_url = st.text_input("Enter Spotify Track URL:")

if st.button("Predict Track"):
    if spotify_url:
        # Extract track ID from Spotify URL
        track_id = spotify_url.split('/')[-1].split('?')[0]

        # Get track features and info
        track = sp.audio_features(track_id)[0]
        track_info = sp.track(track_id)

        # Create DataFrame from track features
        df = pd.DataFrame([{
            'danceability': track['danceability'],
            'duration_ms': track['duration_ms'],
            'energy': track['energy'],
            'instrumentalness': track['instrumentalness'],
            'key': track['key'],
            'liveness': track['liveness'],
            'mode': track['mode'],
            'speechiness': track['speechiness'],
            'tempo': track['tempo'],
            'time_signature': track['time_signature'],
            'song_title': track_info['name'],
            'artist': track_info['artists'][0]['name']
        }])

        # Feature selection (Skipping this as your code already has the required features)
        # df.drop(['acousticness', 'loudness', 'valence'], axis=1, inplace=True, errors='ignore')

        # Handle unknown labels
        df['song_title'] = df['song_title'].apply(lambda x: x if x in known_song_titles else 'Unknown')
        df['artist'] = df['artist'].apply(lambda x: x if x in known_artists else 'Unknown')

        # Label encoding
        df['song_title'] = le_song_title.transform(df['song_title'])
        df['artist'] = le_artist.transform(df['artist'])

        # Scaling
        columns_to_standardize = ['instrumentalness', 'danceability', 'duration_ms', 'energy', 'speechiness', 'liveness', 'tempo']
        df[columns_to_standardize] = scaler.transform(df[columns_to_standardize])

        # Make prediction
        prediction = xgb.predict(df)

        # Display the features
        st.write("## Extracted Features")
        st.dataframe(df)

        # Display the prediction and track info
        st.write(f"**Title:** {track_info['name']}")
        st.write(f"**Artist:** {track_info['artists'][0]['name']}")
        st.write(f"**Predicted Result:** {prediction[0]}")

        # Display visuals based on prediction
        if prediction[0] == 1:
            st.success('I would probably like this song! 🎶👍')
            st.image('https://www.kapwing.com/resources/content/images/2020/04/final_5e94ef5edc305d00159b08e7_769177.gif', use_column_width=True)
        else:
            st.warning('I might not enjoy this song. 😢👎')
            st.image('https://media.tenor.com/qgIjYKSCYfAAAAAC/gordon-ramsey-i-dont.gif', use_column_width=True)

####################################################### SINGLE TRACK PREDICTION END ###########################################################

####################################################### Playlist TRACK PREDICTION Start ###########################################################
# Spotify API setup
client_id = '6291dd76ffb744299366b81b9cddfe5c'
client_secret = '14e1d9b1399a4d8e8ea34e09717b4815'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

st.header("Predict Playlist 📖")
# Text box for Playlist URL
playlist_url = st.text_input("Enter Spotify Playlist URL:")

# Initialize an empty list to store track IDs and an empty dataframe for features
track_ids = []
tracks_df = pd.DataFrame()

if st.button("Predict Playlist"):
    # Step 1: Extract Playlist ID from URL
    if playlist_url:
        pattern = r'https://open\.spotify\.com/playlist/(\w+)'
        match = re.search(pattern, playlist_url)
        if match:
            playlist_id = match.group(1)

            # Step 2: Fetch Track IDs from Playlist
            results = sp.playlist_tracks(playlist_id)
            tracks = results['items']
            for track in tracks:
                track_ids.append(track['track']['id'])

            # Step 3: Retrieve Features for Each Track
            for track_id in track_ids:
                track = sp.audio_features(track_id)[0]
                track_info = sp.track(track_id)
                track_data = {
                    'danceability': track['danceability'],
                    'duration_ms': track['duration_ms'],
                    'energy': track['energy'],
                    'instrumentalness': track['instrumentalness'],
                    'key': track['key'],
                    'liveness': track['liveness'],
                    'mode': track['mode'],
                    'speechiness': track['speechiness'],
                    'tempo': track['tempo'],
                    'time_signature': track['time_signature'],
                    'song_title': track_info['name'],
                    'artist': track_info['artists'][0]['name']
                }
                tracks_df = pd.concat([tracks_df, pd.DataFrame([track_data])], ignore_index=True)


            # Step 4: Preprocess the Features
            tracks_df['song_title'] = tracks_df['song_title'].apply(lambda x: x if x in known_song_titles else 'Unknown')
            tracks_df['artist'] = tracks_df['artist'].apply(lambda x: x if x in known_artists else 'Unknown')
            tracks_df['song_title'] = le_song_title.transform(tracks_df['song_title'])
            tracks_df['artist'] = le_artist.transform(tracks_df['artist'])
            columns_to_standardize = ['instrumentalness', 'danceability', 'duration_ms', 'energy', 'speechiness', 'liveness', 'tempo']
            tracks_df[columns_to_standardize] = scaler.transform(tracks_df[columns_to_standardize])

            # Step 5: Make Predictions
            predictions = xgb.predict(tracks_df)

            # Display the total counts
            count_liked = sum(predictions == 1)
            count_disliked = sum(predictions == 0)

            # Create columns for text and image
            col1, col2 = st.columns([1, 1])

            # First column: Total number of Liked songs and an image
            with col1:
                # Using Markdown to center-align the text, add color, and an emoji
                st.markdown(
                    f"<div style='text-align: center; color: green;'><h2>👍 Liked Songs: {count_liked}</h2></div>",
                    unsafe_allow_html=True,
                )
                # Display an image
                # Center-aligning by using column layout
                col1_1, col1_2, col1_3 = st.columns([1, 1, 1])
                with col1_2:
                    st.image("https://www.trustedreviews.com/wp-content/uploads/sites/54/2021/02/Rickrolling-in-4K.jpg", width=150)

            # Second column: Total number of Disliked songs and an image
            with col2:
                # Using Markdown to center-align the text, add color, and an emoji
                st.markdown(
                    f"<div style='text-align: center; color: red;'><h2>👎 Disliked Songs: {count_disliked}</h2></div>",
                    unsafe_allow_html=True,
                )
                # Display an image
                # Center-aligning by using column layout
                col2_1, col2_2, col2_3 = st.columns([1, 1, 1])
                with col2_2:
                    st.image("https://i.guim.co.uk/img/media/cf19bda612430e6ff33122df8cec403700a38403/794_471_907_544/500.jpg?quality=85&auto=format&fit=max&s=9f3a32b8c2fd3cad2231cc164080ee5a", width=150)


            # Step 6: Display Predictions
            # Create a subheader for the predictions section
            st.subheader("Song Predictions")

            # Initialize an empty string to store the Markdown content
            markdown_text = "| Title | Artist | Prediction |\n| --- | --- | --- |\n"

            # Loop through each prediction and append to the Markdown content
            for i, pred in enumerate(predictions):
                song_title = tracks[i]['track']['name']
                artist = tracks[i]['track']['artists'][0]['name']
                like_status = '<span style="color: green;">Like</span>' if pred == 1 else '<span style="color: red;">Dislike</span>'
                markdown_text += f"| {song_title} | {artist} | {like_status} |\n"

            # Display the Markdown content as a table
            st.markdown(markdown_text, unsafe_allow_html=True)


####################################################### Playlist TRACK PREDICTION End ###########################################################

