"""Utilities for generating and formatting recommendations."""

from typing import Dict, Optional, Tuple
import pandas as pd
from System.recommendation.utils.user_analysis import analyze_user_listening_history
import logging

logger = logging.getLogger(__name__)

def get_recommendations(
    system,
    user_id: str,
    recommender,
    n: int = 10,
    exclude_listened: bool = True,
    include_user_profile: bool = True,
    verbose: bool = True,
    testing_mode: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    """Generate recommendations for a user.
    
    Args:
        system: RecommendationSystem instance
        user_id: Target user ID
        recommender: Recommender instance
        n: Number of recommendations
        exclude_listened: Exclude listened items
        include_user_profile: Include user profile
        verbose: Print progress messages
        testing_mode: Enable detailed logging
        
    Returns:
        Tuple of (recommendations DataFrame, user profile dict)
    """
    if verbose:
        print(f"Generating {n} recommendations for user {user_id} using {recommender.name}...")
    
    recommendations = recommender.recommend(
        user_id=user_id,
        n=n,
        exclude_listened=exclude_listened,
        include_metadata=True,
        testing_mode=testing_mode
    )
    
    user_profile = {}
    if include_user_profile:
        interactions = system.data_manager.user_loader.get_interactions()
        user_items = interactions[interactions['user_id'] == user_id]
        if not user_items.empty:
            user_profile = analyze_user_listening_history(user_items, system.data_manager.song_loader.get_song_metadata())
            if verbose and testing_mode:
                print(f"\n=== USER LISTENING PROFILE FOR USER {user_id} ===")
                print(user_profile['formatted'])
    if testing_mode:
        logger.info("Generated %d recommendations for user %s", len(recommendations), user_id)
    return recommendations, user_profile

def get_similar_items(
    system,
    seed_item_id: str,
    recommender,
    n: int = 10,
    exclude_seed: bool = True,
    verbose: bool = True,
    user_id: Optional[str] = None,
    testing_mode: bool = False
) -> pd.DataFrame:
    """Generate similar item recommendations.
    
    Args:
        system: RecommendationSystem instance
        seed_item_id: Seed item ID
        recommender: Recommender instance
        n: Number of recommendations
        exclude_seed: Exclude seed item
        verbose: Print progress messages
        user_id: Optional user ID for personalization
        testing_mode: Enable detailed logging
        
    Returns:
        Recommendations DataFrame
    """
    if verbose:
        print(f"Generating {n} similar items for seed item {seed_item_id} using {recommender.name}...")
    
    recommendations = recommender.recommend_similar_items(
        seed_item_id=seed_item_id,
        n=n,
        exclude_seed=exclude_seed,
        user_id=user_id,
        testing_mode=testing_mode
    )
    
    logger.info("Generated %d similar items for seed %s", len(recommendations), seed_item_id)
    return recommendations