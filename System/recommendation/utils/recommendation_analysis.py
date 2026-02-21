"""Utilities for analyzing and comparing recommendations."""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
from System.recommendation.utils.display import print_recommendation_details
import logging

logger = logging.getLogger(__name__)

def analyze_recommendations(
    user_id: str,
    recommendations: pd.DataFrame,
    user_profile: Dict[str, Any]
) -> Dict[str, Any]:
    results = {}
    
    if 'top_genre' in recommendations.columns and user_profile.get('top_genres'):
        user_genres = set(user_profile['top_genres'])
        user_all_genres = set(user_profile.get('all_genres', []))
        rec_genres = set(recommendations['top_genre'].dropna())
        
        top_genre_overlap = user_genres.intersection(rec_genres)
        top_overlap_pct = len(top_genre_overlap) / len(user_genres) * 100 if user_genres else 0
        all_genre_overlap = user_all_genres.intersection(rec_genres)
        all_overlap_pct = len(all_genre_overlap) / len(user_all_genres) * 100 if user_all_genres else 0
        matching_recs = recommendations[recommendations['top_genre'].isin(user_all_genres)]
        taste_match_pct = len(matching_recs) / len(recommendations) * 100 if len(recommendations) > 0 else 0
        
        results.update({
            'top_genre_overlap': top_genre_overlap,
            'top_overlap_pct': top_overlap_pct,
            'all_genre_overlap': all_genre_overlap,
            'all_overlap_pct': all_overlap_pct,
            'taste_match_pct': taste_match_pct,
            'matching_recs': len(matching_recs),
            'new_genre_recs': len(recommendations) - len(matching_recs)
        })
    
    if 'artist_tier' in recommendations.columns:
        tier_counts = recommendations['artist_tier'].value_counts()
        total_recs = len(recommendations)
        tier_distribution = {
            tier: {'count': count, 'percentage': count / total_recs * 100}
            for tier, count in tier_counts.items()
        }
        tier_diversity = 1 - (tier_counts.max() / total_recs if total_recs > 0 else 0)
        
        results.update({
            'tier_distribution': tier_distribution,
            'tier_diversity': tier_diversity
        })
    
    logger.info("Analyzed recommendations for user %s", user_id)
    return results

def compare_recommendation_approaches(
    system,
    user_id: str,
    seed_item_id: str,
    recommender, # This is still the recommender object
    n: int = 10,
    user_item_recs: Optional[pd.DataFrame] = None,
    user_profile: Optional[Dict[str, Any]] = None,
    testing_mode: bool = True,
    suppress_language_distribution: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    logger.info("Comparing recommendation approaches for user %s and seed %s using %s", user_id, seed_item_id, recommender.name)
    print(f"\n=== RECOMMENDATION APPROACH COMPARISON ===")
    print(f"User ID: {user_id}, Recommender: {recommender.name}") # Use recommender.name

    seed_song_info = system.data_manager.song_loader.get_song_metadata()
    seed_song_info = seed_song_info[seed_song_info['song_id'] == seed_item_id]
    if not seed_song_info.empty:
        title_col = 'title' if 'title' in seed_song_info.columns else None
        artist_col = next((col for col in ['artist_name', 'artist'] if col in seed_song_info.columns), None)
        genre_col = 'top_genre' if 'top_genre' in seed_song_info.columns else None
        if title_col and artist_col:
            print(f"Seed Song: \"{seed_song_info[title_col].iloc[0]}\" by {seed_song_info[artist_col].iloc[0]}")
        if genre_col:
            print(f"Seed Song Genre: {seed_song_info[genre_col].iloc[0]}")
    
    if user_item_recs is None:
        print("\n1. GENERATING USER-BASED RECOMMENDATIONS")
        user_item_recs, user_profile = system.get_recommendations(
            user_id=user_id,
            recommender_name=recommender.name,
            n=n,
            include_user_profile=True,
            verbose=True,
            testing_mode=testing_mode
        )
    else:
        print("\n1. USING PRE-COMPUTED USER-BASED RECOMMENDATIONS")
    
    if testing_mode:
        print_recommendation_details(user_item_recs, header="USER-BASED RECOMMENDATIONS")
    
    print("\n2. GENERATING ITEM-BASED RECOMMENDATIONS")
    item_item_recs = system.get_similar_items(
        seed_item_id=seed_item_id,
        recommender_name=recommender.name,
        n=n,
        verbose=True,
        # Removed user_id as it's not expected by system.get_similar_items
    )
    
    if testing_mode:
        print_recommendation_details(item_item_recs, header="ITEM-BASED RECOMMENDATIONS")
    
    user_item_id_col = next((col for col in ['song_id', 'msd_song_id'] if col in user_item_recs.columns), None)
    item_item_id_col = next((col for col in ['song_id', 'msd_song_id'] if col in item_item_recs.columns), None)
    
    if user_item_id_col and item_item_id_col:
        user_item_songs = set(user_item_recs[user_item_id_col])
        item_item_songs = set(item_item_recs[item_item_id_col])
        overlap = user_item_songs.intersection(item_item_songs)
        
        print(f"\n=== COMPARISON RESULTS ===")
        print(f"Overlap: {len(overlap)} songs ({len(overlap)/n*100:.1f}%)")
        
        if overlap and 'title' in user_item_recs.columns and 'artist_name' in user_item_recs.columns:
            print("\nOverlapping Songs:")
            overlap_songs = user_item_recs[user_item_recs[user_item_id_col].isin(overlap)]
            for _, row in overlap_songs.iterrows():
                print(f"• {row['title']} by {row['artist_name']}")
        
        if 'top_genre' in user_item_recs.columns and 'top_genre' in item_item_recs.columns:
            user_genres = user_item_recs['top_genre'].value_counts().to_dict()
            item_genres = item_item_recs['top_genre'].value_counts().to_dict()
            all_genres = sorted(set(user_genres.keys()) | set(item_genres.keys()))
            
            print("\nGenre Distribution Comparison:")
            print(f"{'Genre':<20} {'User-Based':<10} {'Item-Based':<10}")
            print("-" * 45)
            for genre in all_genres:
                user_count = user_genres.get(genre, 0)
                item_count = item_genres.get(genre, 0)
                print(f"{genre:<20} {user_count:<10} {item_count:<10}")
        
        if 'artist_tier' in user_item_recs.columns and 'artist_tier' in item_item_recs.columns:
            user_tiers = user_item_recs['artist_tier'].value_counts().to_dict()
            item_tiers = item_item_recs['artist_tier'].value_counts().to_dict()
            all_tiers = sorted(set(user_tiers.keys()) | set(item_tiers.keys()))
            
            print("\nArtist Tier Distribution Comparison:")
            print(f"{'Tier':<20} {'User-Based':<10} {'Item-Based':<10}")
            print("-" * 45)
            for tier in all_tiers:
                user_count = user_tiers.get(tier, 0)
                item_count = item_tiers.get(tier, 0)
                print(f"{tier:<20} {user_count:<10} {item_count:<10}")
    
    if not suppress_language_distribution and testing_mode and 'language' in user_item_recs.columns and 'language' in item_item_recs.columns:
        print("\n=== LANGUAGE DISTRIBUTION ===")
        print("User-Item Recommendations:")
        lang_counts = user_item_recs['language'].value_counts()
        for lang, count in lang_counts.items():
            print(f"• {lang}: {count} songs ({count/len(user_item_recs)*100:.1f}%)")
        
        print("\nItem-Item Recommendations:")
        lang_counts = item_item_recs['language'].value_counts()
        for lang, count in lang_counts.items():
            print(f"• {lang}: {count} songs ({count/len(item_item_recs)*100:.1f}%)")
    
    logger.info("Recommendation comparison complete")
    return user_item_recs, item_item_recs

