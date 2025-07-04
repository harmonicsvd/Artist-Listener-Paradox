"""Utilities for analyzing user listening history."""

from typing import Dict
import pandas as pd
from collections import Counter

def analyze_user_listening_history(
    user_interactions: pd.DataFrame,
    song_metadata: pd.DataFrame
) -> Dict:
    """Analyze a user's listening history to extract profile statistics."""
    profile = {}
    
    # Debug: Log input data
    #print(f"song_metadata columns: {song_metadata.columns.tolist()}")
    #print(f"user_interactions columns: {user_interactions.columns.tolist()}")
    #print(f"user_interactions shape: {user_interactions.shape}")
    
    # Basic statistics
    profile['total_listens'] = len(user_interactions)
    profile['unique_songs'] = user_interactions['song_id'].nunique()
    
    # Identify song_id column in song_metadata
    possible_id_cols = ['song_id', 'track_id', 'id']
    id_col = next((col for col in possible_id_cols if col in song_metadata.columns), song_metadata.columns[0])
    #print(f"Using song_id column: {id_col}")
    
    # Create mappings from song_metadata
    song_to_artist = dict(zip(song_metadata[id_col], song_metadata.get('artist_name', pd.Series(index=song_metadata[id_col], dtype=str))))
    song_to_genre = dict(zip(song_metadata[id_col], song_metadata.get('top_genre', pd.Series(index=song_metadata[id_col], dtype=str))))
    song_to_language = dict(zip(song_metadata[id_col], song_metadata.get('language', pd.Series(index=song_metadata[id_col], dtype=str))))
    
    # Language analysis
    languages = []
    
    # First, try user_interactions['language']
    if 'language' in user_interactions.columns:
        language_counts = user_interactions['language'].fillna('Unknown').value_counts()
        if language_counts.sum() > 0:
            languages = user_interactions['language'].fillna('Unknown').tolist()
            #print(f"Using language from user_interactions: {language_counts.to_dict()}")
    
    # Fallback to song_metadata language_ columns or language column
    if not languages:
        language_columns = [col for col in song_metadata.columns if col.startswith('language_')]
        if language_columns:
            for song_id in user_interactions['song_id']:
                song_data = song_metadata[song_metadata[id_col] == song_id]
                if not song_data.empty:
                    song_languages = [
                        col.replace('language_', '') for col in language_columns
                        if pd.notna(song_data[col].iloc[0]) and song_data[col].iloc[0] == 1
                    ]
                    languages.append(song_languages[0] if song_languages else 'Unknown')
                else:
                    languages.append('Unknown')
                    print(f"Warning: Song {song_id} not found in song_metadata")
            #print(f"Using language_ columns from song_metadata: {language_columns}")
        elif 'language' in song_metadata.columns:
            languages = user_interactions['song_id'].map(song_to_language).fillna('Unknown').tolist()
            #print(f"Using language column from song_metadata: {pd.Series(languages).value_counts().to_dict()}")
        else:
            languages = ['Unknown'] * len(user_interactions)
            print("Warning: No 'language' in user_interactions or 'language_'/'language' columns in song_metadata")
    
    # Compute language distribution
    language_counts = pd.Series(languages).value_counts()
    profile['all_languages'] = language_counts.index.tolist() or ['Unknown']
    profile['language_distribution'] = {
        lang: f"{count} songs ({count/len(user_interactions)*100:.1f}%)"
        for lang, count in language_counts.items()
    }
    if not profile['all_languages']:
        profile['all_languages'] = ['Unknown']
        profile['language_distribution'] = {'Unknown': f"0 songs (0.0%)"}
    #print(f"Language counts: {language_counts.to_dict()}")
    
    # Artist analysis
    artists = user_interactions['song_id'].map(song_to_artist).fillna('Unknown')
    artist_counts = artists.value_counts()
    profile['artist_diversity'] = len(artist_counts) / profile['unique_songs'] if profile['unique_songs'] > 0 else 0
    profile['top_artists'] = artist_counts.head(2).index.tolist()
    if artist_counts.get('Unknown', 0) > 0:
        print(f"Warning: {artist_counts['Unknown']} songs with unknown artist")
    
    # Genre analysis
    genres = user_interactions['song_id'].map(song_to_genre).fillna('Unknown')
    genre_counts = genres.value_counts()
    profile['all_genres'] = genre_counts.index.tolist()
    profile['genre_diversity'] = len(genre_counts) / profile['unique_songs'] if profile['unique_songs'] > 0 else 0
    profile['top_genres'] = genre_counts.head(2).index.tolist()
    if genre_counts.get('Unknown', 0) > 0:
        print(f"Warning: {genre_counts['Unknown']} songs with unknown genre")
    
    # Format output
    output = []
    output.append(f"Total Listens: {profile['total_listens']}")
    output.append(f"Unique Songs: {profile['unique_songs']}")
    output.append(f"Artist Diversity: {profile['artist_diversity']:.2f}")
    output.append(f"Genre Diversity: {profile['genre_diversity']:.2f}")
    output.append(f"Top Artists: {', '.join(profile['top_artists']) if profile['top_artists'] else 'Unknown'}")
    output.append(f"All Genres: {', '.join(profile['all_genres']) if profile['all_genres'] else 'Unknown'}")
    output.append(f"Top Genres: {', '.join(profile['top_genres']) if profile['top_genres'] else 'Unknown'}")
    output.append(f"All Languages: {', '.join(profile['all_languages']) if profile['all_languages'] else 'Unknown'}")
    output.append("\nLanguage Distribution in User's Listening History:")
    for lang, dist in profile['language_distribution'].items():
        output.append(f"• {lang}: {dist}")
    
    profile['formatted'] = '\n'.join(output)
    return profile