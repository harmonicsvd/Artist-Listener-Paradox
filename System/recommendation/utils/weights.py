from typing import Dict, List
import logging
import numpy as np

logger = logging.getLogger(__name__)

def calculate_language_weights(
    item_ids: List[str],
    play_counts: List[float],
    song_to_language: Dict[str, str]
) -> Dict[str, float]:
    """Calculate weights for languages based on user's listening history.
    
    Args:
        item_ids: List of item IDs
        play_counts: List of play counts
        song_to_language: Mapping from song IDs to languages
        
    Returns:
        Dictionary mapping languages to weights
    """
    language_counts = {}
    total_plays = sum(play_counts)
    missing_languages = 0
    
    for item_id, play_count in zip(item_ids, play_counts):
        if item_id in song_to_language:
            language = song_to_language[item_id]
            language_counts[language] = language_counts.get(language, 0) + play_count
        else:
            missing_languages += 1
    
    if missing_languages > 0:
        logger.warning(f"Missing language for {missing_languages} songs")
    
    language_weights = {}
    if total_plays > 0:
        total_weight = 0.0
        for language, count in language_counts.items():
            weight = count / total_plays
            language_weights[language] = max(0.2, 0.7 * np.log1p(2.0 * weight))
            total_weight += language_weights[language]
        
        # Normalize to sum to 1.0
        if total_weight > 0:
            for language in language_weights:
                language_weights[language] /= total_weight
    
    return language_weights

def calculate_genre_weights(
    item_ids: List[str],
    play_counts: List[float],
    song_to_genres: Dict[str, List[str]]
) -> Dict[str, float]:
    """Calculate weights for genres based on user's listening history.
    
    Args:
        item_ids: List of item IDs
        play_counts: List of play counts
        song_to_genres: Mapping from song IDs to genre tags
        
    Returns:
        Dictionary mapping genres to weights
    """
    genre_counts = {}
    total_plays = sum(play_counts)
    missing_genres = 0
    
    for item_id, play_count in zip(item_ids, play_counts):
        if item_id in song_to_genres:
            genres = song_to_genres[item_id]
            if genres:
                for i, genre in enumerate(genres):
                    weight = play_count * (2.0 if i == 0 else 0.3)  # Adjusted from 2.5/0.2
                    genre_counts[genre] = genre_counts.get(genre, 0) + weight
            else:
                missing_genres += 1
        else:
            missing_genres += 1
    
    if missing_genres > 0:
        logger.warning(f"Missing genres for {missing_genres} songs")
    
    genre_weights = {}
    if total_plays > 0:
        total_weight = 0.0
        for genre, count in genre_counts.items():
            weight = count / total_plays
            genre_weights[genre] = max(0.2, 0.7 * np.log1p(2.0 * weight))
            total_weight += genre_weights[genre]
        
        # Normalize to sum to 1.0
        if total_weight > 0:
            for genre in genre_weights:
                genre_weights[genre] /= total_weight
    
    return genre_weights

def calculate_item_weights(
    item_id: str,
    song_to_tier: Dict[str, str],
    song_to_pop: Dict[str, float],
    song_to_fam: Dict[str, float],
    genre_weights: Dict[str, float],
    song_to_genres: Dict[str, List[str]],
    song_to_language: Dict[str, str],
    language_weights: Dict[str, float],
    tier_weights: Dict[str, float]
) -> float:
    weight = 1.0

    # Tier weight (no amplification or penalties)
    tier = song_to_tier.get(item_id, 'mid_tier')
    tier_weight = tier_weights.get(tier, 1.0)
    weight *= tier_weight

    # Genre weight with similarity filter
    genres = song_to_genres.get(item_id, [])
    if genres:
        genre_weight = sum(genre_weights.get(genre, 0.0) * (2.0 if i == 0 else 0.3) for i, genre in enumerate(genres))  # Adjusted from 2.0/0.1
        genre_weight /= sum(2.0 if i == 0 else 0.3 for i in range(len(genres)))
        # Penalize items with no genre overlap
        item_genres = set(genres)
        user_genres = set(genre_weights.keys())
        if not item_genres.intersection(user_genres):
            genre_weight *= 0.01
        weight *= max(0.1, 1.0 + genre_weight)  # Kept floor at 0.1
    else:
        weight *= 0.1  # Penalize items with no genres

    # Language weight
    language = song_to_language.get(item_id)
    if language and language in language_weights:
        weight *= max(0.5, 1.0 + 2.0 * language_weights[language])
    else:
        weight *= 0.2  # Penalize unknown languages

    # Cap weight
    weight = min(weight, 3.0)
    return max(0.1, weight)