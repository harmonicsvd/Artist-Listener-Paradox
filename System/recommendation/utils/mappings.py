from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

def compute_related_genres(song_to_genres: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Compute related genres based on co-occurrence in song_to_genres.
    
    Args:
        song_to_genres: Mapping from song_id to list of genres.
    
    Returns:
        Dictionary mapping each genre to a list of related genres.
    """
    genre_cooccurrence = defaultdict(set)
    for genres in song_to_genres.values():
        for i, genre1 in enumerate(genres):
            for genre2 in genres[i+1:]:
                genre_cooccurrence[genre1].add(genre2)
                genre_cooccurrence[genre2].add(genre1)
                
    return {genre: sorted(list(related)) for genre, related in genre_cooccurrence.items()}

def get_related_genres(genre: str, song_to_genres: Dict[str, List[str]]) -> List[str]:
    """Return a list of genres related to the input genre based on co-occurrence.
    
    Args:
        genre: The input genre (e.g., 'blues-rock').
        song_to_genres: Mapping from song_id to list of genres.
    
    Returns:
        List of related genres, including the input genre as fallback.
    """
    related_genres_map = compute_related_genres(song_to_genres)
    related = related_genres_map.get(genre, [])
    print(f"Related genres for '{genre}': {related}")
    # Fallback to input genre to ensure non-empty recommendations
    return related + [genre] if genre not in related else related

def create_song_to_tier_mapping(song_metadata: Optional[pd.DataFrame]) -> Dict[str, str]:
    """Create a mapping from song IDs to artist tiers.
    
    Args:
        song_metadata: DataFrame containing song metadata with artist tier information
        
    Returns:
        Dictionary mapping song IDs to artist tiers
    """
    song_to_tier = {}
    
    if song_metadata is not None and 'artist_tier' in song_metadata.columns:
        id_col = next((col for col in ['song_id', 'track_id', 'id'] if col in song_metadata.columns), song_metadata.columns[0])
        song_to_tier = dict(zip(song_metadata[id_col], song_metadata['artist_tier'].fillna('Unknown')))
        logger.info("Created tier mapping for %d songs", len(song_to_tier))
    
    return song_to_tier

def create_artist_metric_mappings(
    song_metadata: Optional[pd.DataFrame],
    artist_identification: Optional[pd.DataFrame]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Create mappings from song IDs to artist popularity and familiarity.
    
    Args:
        song_metadata: DataFrame containing song metadata with artist IDs
        artist_identification: DataFrame containing artist metrics
        
    Returns:
        Tuple of dictionaries mapping song IDs to popularity and familiarity
    """
    song_to_pop = {}
    song_to_fam = {}
    
    if song_metadata is not None and artist_identification is not None and 'artist_id' in song_metadata.columns:
        id_col = next((col for col in ['song_id', 'track_id', 'id'] if col in song_metadata.columns), song_metadata.columns[0])
        
        artist_to_pop = dict(zip(
            artist_identification['artist_id'],
            artist_identification.get('external_popularity', pd.Series(index=artist_identification['artist_id'], dtype=float))
        ))
        artist_to_fam = dict(zip(
            artist_identification['artist_id'],
            artist_identification.get('external_familiarity', pd.Series(index=artist_identification['artist_id'], dtype=float))
        ))
        
        for _, row in song_metadata.iterrows():
            song_id = row[id_col]
            artist_id = row['artist_id']
            if pd.notna(artist_id):
                song_to_pop[song_id] = artist_to_pop.get(artist_id, 0.0)
                song_to_fam[song_id] = artist_to_fam.get(artist_id, 0.0)
        
        logger.info("Created popularity mapping for %d songs", len(song_to_pop))
        logger.info("Created familiarity mapping for %d songs", len(song_to_fam))
    
    return song_to_pop, song_to_fam

def create_genre_mapping(song_metadata: pd.DataFrame) -> Dict[str, List[str]]:
    """Create mapping from song_id to list of genres, prioritizing top_genre and co-occurring genres."""
    song_to_genres = {}
    id_col = next((col for col in ['song_id', 'track_id', 'id'] if col in song_metadata.columns), song_metadata.columns[0])
    
    # Get genre columns
    genre_columns = [col for col in song_metadata.columns if col.startswith('genre_')]
    if not genre_columns:
        logger.warning("No genre columns found in song metadata")
        return {}
    
    # Check if top_genre exists
    has_top_genre = 'top_genre' in song_metadata.columns
    
    # Compute genre co-occurrence for relevance
    genre_cooccurrence = defaultdict(lambda: defaultdict(int))
    for _, row in song_metadata.iterrows():
        if has_top_genre and pd.notna(row['top_genre']):
            top_genre = row['top_genre']
            for col in genre_columns:
                if row[col] == 1:
                    genre = col.replace('genre_', '')
                    if genre != top_genre:
                        genre_cooccurrence[top_genre][genre] += 1
    
    for _, row in song_metadata.iterrows():
        song_id = row[id_col]
        genres = []
        
        # Add top_genre if available and valid
        if has_top_genre and pd.notna(row['top_genre']):
            genres.append(row['top_genre'])
        
        # Get other genres where value is 1
        other_genres = [col.replace('genre_', '') for col in genre_columns if row[col] == 1 and (not has_top_genre or col.replace('genre_', '') != row.get('top_genre', ''))]
        
        # Sort other genres by co-occurrence with top_genre (if available)
        if has_top_genre and pd.notna(row['top_genre']):
            top_genre = row['top_genre']
            other_genres.sort(key=lambda g: genre_cooccurrence[top_genre].get(g, 0), reverse=True)
        
        # Add up to 4 other genres
        genres.extend(other_genres[:4])
        
        # Cap genres at 5
        genres = genres[:5]
        
        # Fallback if no genres
        song_to_genres[song_id] = genres if genres else ['Unknown']
        if len(genres) > 5:
            logger.warning(f"Song {song_id} has {len(genres)} genres after capping: {genres}")
        elif len(genres) > 1:
            logger.debug(f"Song {song_id} genres: {genres}")
    
    logger.info("Created genre mapping for %d songs", len(song_to_genres))
    return song_to_genres

def create_language_mapping(song_metadata: pd.DataFrame) -> Dict[str, str]:
    """Create mapping from song_id to language."""
    language_columns = [col for col in song_metadata.columns if col.startswith('language_')]
    if not language_columns:
        logger.warning("No language columns found in song metadata")
        return {}
    
    id_col = next((col for col in ['song_id', 'track_id', 'id'] if col in song_metadata.columns), song_metadata.columns[0])
    song_to_language = {}
    for _, row in song_metadata.iterrows():
        languages = [col.replace('language_', '').capitalize() for col in language_columns if row[col] == 1]
        song_to_language[row[id_col]] = languages[0] if languages else 'Unknown'
    logger.info("Created language mapping for %d songs", len(song_to_language))
    return song_to_language