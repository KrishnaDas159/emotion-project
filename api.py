from dotenv import load_dotenv
import os
import base64
from requests import post, get
import json
import urllib.parse  # <-- Added for URL encoding

# Load environment variables
load_dotenv()

# Spotify API credentials
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

# Emotion-to-Music Mapping
emotion_dict = {0: "sad", 1: "fear", 2: "surprise", 3: "neutral", 4: "disgust", 5: "happy", 6: "angry"}
music_dist = {
    0: "1n6cpWo9ant4WguEo91KZh",  # Sad
    1: "4cllEPvFdoX6NIVWPKai9I",  # Fearful
    2: "4amvTpvtGLi5Vybl1bEDWV",  # surprised
    3: "4kvSlabrnfRCQWfN0MgtgA",  # Neutral
    4: "1n6cpWo9ant4WguEo91KZh",  # Disgusted
    5: "0deORnapZgrxFY4nsKr9JA",  # Happy
    6: "0l9dAmBrUJLylii66JOsHB",  # Angry
}

def get_token():
    """Get Spotify access token dynamically"""
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")
    
    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    
    result = post(url, headers=headers, data=data)

    # Debugging Response
    if result.status_code != 200:
        print(f"❌ Failed Request (Status {result.status_code}):", result.text)
        return None

    json_result = json.loads(result.content)
    token = json_result.get("access_token")

    if not token:
        print("❌ Token retrieval failed. Response:", json_result)
        return None
    
    return token

def get_track_details(playlist_id, token):
    """Fetch track details from a Spotify playlist and add YouTube URL"""
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    headers = {"Authorization": f"Bearer {token}"}
    
    response = get(url, headers=headers)
    if response.status_code != 200:
        print(f"❌ Failed to fetch playlist (Status {response.status_code}):", response.text)
        return []
    
    data = response.json()
    tracks = []
    for item in data['items']:
        track = item['track']
        # Construct a YouTube search URL for the official video
        query = f"{track['name']} {track['artists'][0]['name']} official video"
        youtube_url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)
        
        tracks.append({
            "name": track['name'],
            "artist": track['artists'][0]['name'],
            "id": track['id'],
            "youtube_url": youtube_url  # <-- Added YouTube URL for the track
        })
    return tracks

def recommend_music(emotion_id):
    """Fetch music recommendations based on emotion and return track details."""
    token = get_token()
    if not token:
        print("❌ Failed to get Spotify token.")
        return None

    playlist_id = music_dist.get(emotion_id, music_dist[4])  # Default to neutral if emotion_id is invalid
    print(f"🎧 Fetching {emotion_dict[emotion_id]} playlist...")

    tracks = get_track_details(playlist_id, token)
    if not tracks:
        print("❌ No tracks found in the playlist.")
        return None

    print(f"\n🎵 Recommended {emotion_dict[emotion_id]} Playlist:")
    for i, track in enumerate(tracks[:10], 1):
        print(f"{i}. {track['name']} - {track['artist']}")
    return tracks[:10]

# Example usage
if __name__ == "__main__":
    # Test with a specific emotion (e.g., Happy)
    emotion_id = 0  # Happy
    recommend_music(emotion_id)
