from typing import Dict, List
import logging
import numpy as np

logger = logging.getLogger(__name__)

def calculate_language_weights(
    item_ids: List[str],
    play_counts: List[float],
    song_to_language: Dict[str, str]
) -> Dict[str, float]:
    """
    Calculates the proportion of user's listening history per language.
    This function's output is now primarily for analytical purposes or to determine a user's dominant language for filtering,
    rather than being used as a direct multiplicative weight in item scoring.
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
        logger.warning(f"Missing language for {missing_languages} songs when calculating language counts.")
    
    language_proportions = {}
    if total_plays > 0:
        for language, count in language_counts.items():
            language_proportions[language] = count / total_plays
    
    return language_proportions

def calculate_genre_weights(
    item_ids: List[str],
    play_counts: List[float],
    song_to_genres: Dict[str, List[str]]
) -> Dict[str, float]:
    """Calculate weights for genres based on user's listening history."""
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
        
        # Normalize weights to sum to 1.0
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
    song_to_language: Dict[str, str], # Still passed for context, but not for direct weighting
    language_weights: Dict[str, float], # This parameter is now effectively ignored for scoring
    tier_weights: Dict[str, float]
) -> float:
    weight = 1.0

    # Tier weight
    tier = song_to_tier.get(item_id, 'mid_tier')
    tier_weight = tier_weights.get(tier, 1.0)
    weight *= tier_weight

    # Genre weight with similarity filter
    genres = song_to_genres.get(item_id, [])
    if genres:
        genre_score_sum = 0.0
        total_genre_contribution = 0.0
        for i, genre in enumerate(genres):
            user_preference_for_genre = genre_weights.get(genre, 0.0)
            contribution_factor = 2.0 if i == 0 else 0.3 
            genre_score_sum += user_preference_for_genre * contribution_factor
            total_genre_contribution += contribution_factor
        
        if total_genre_contribution > 0:
            genre_weight = genre_score_sum / total_genre_contribution
        else:
            genre_weight = 0.0

        item_genres_set = set(genres)
        user_preferred_genres_set = set(genre_weights.keys())
        if not item_genres_set.intersection(user_preferred_genres_set):
            genre_weight *= 0.01 
        
        weight *= max(0.1, 1.0 + genre_weight) 
    else:
        weight *= 0.8 # Penalty for items without genre info

    # Language weight is now handled by direct filtering in the recommender's `recommend` method.
    # Therefore, we explicitly remove its influence from `calculate_item_weights`.
    # The `language_weights` parameter is accepted for compatibility but its value is not used.
    # No code here to multiply `weight` by `language_weights`.

    # Popularity and Familiarity weights
    pop_score = song_to_pop.get(item_id, 0.0)
    fam_score = song_to_fam.get(item_id, 0.0)

    # Apply a gentle boost for higher popularity/familiarity, but not too much to avoid bias
    weight *= (1.0 + pop_score * 0.5 + fam_score * 0.5) 

    # Cap weight (optional, but good for stability)
    weight = min(weight, 3.0) # Example cap
    return max(0.1, weight) # Ensure a minimum overall weight
