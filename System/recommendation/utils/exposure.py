from typing import Dict, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def track_exposure(
    recommendations: pd.DataFrame,
    song_to_artist: Optional[Dict[str, str]] = None,
    artist_exposure: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    """Track artist exposure from recommendations.
    
    Args:
        recommendations: DataFrame containing recommendations
        song_to_artist: Mapping from song IDs to artist names
        artist_exposure: Dictionary tracking artist exposure counts
        
    Returns:
        DataFrame with current exposure counts
    """
    if artist_exposure is None:
        artist_exposure = {}
    
    if 'artist_name' in recommendations.columns:
        for _, row in recommendations.iterrows():
            artist = row['artist_name']
            if pd.notna(artist):
                artist_exposure[artist] = artist_exposure.get(artist, 0) + 1
    elif song_to_artist and 'song_id' in recommendations.columns:
        for song_id in recommendations['song_id']:
            if artist := song_to_artist.get(song_id):
                artist_exposure[artist] = artist_exposure.get(artist, 0) + 1
    
    exposure_df = pd.DataFrame(
        list(artist_exposure.items()),
        columns=['artist_name', 'exposure_count']
    )
    #logger.info("Tracked exposure for %d artists", len(exposure_df))
    return exposure_df

def analyze_exposure_distribution(
    artist_exposure: Dict[str, int],
    song_metadata: Optional[pd.DataFrame] = None
) -> Dict:
    """Analyze exposure across artist tiers with detailed statistics.
    
    Args:
        artist_exposure: Dictionary tracking artist exposure counts
        song_metadata: DataFrame containing song metadata with artist tier information
        
    Returns:
        Dict containing exposure_df (DataFrame), tier_diversity (float), and gini_coefficient (float)
    """
    if not artist_exposure:
        logger.warning("No exposure data available")
        print("No exposure data available yet")
        return {'exposure_df': pd.DataFrame(), 'tier_diversity': 0.0, 'gini_coefficient': 0.0}
    
    exposure_df = pd.DataFrame(
        list(artist_exposure.items()),
        columns=['artist_name', 'exposure_count']
    )
    
    print("\n=== ARTIST EXPOSURE ANALYSIS ===")
    print(f"Total Artists Exposed: {len(exposure_df)}")
    print(f"Total Recommendations: {exposure_df['exposure_count'].sum()}")
    print(f"Average Exposure per Artist: {exposure_df['exposure_count'].mean():.2f}")
    print(f"Maximum Exposure for a Single Artist: {exposure_df['exposure_count'].max()}")
    
    gini = calculate_gini_coefficient(exposure_df['exposure_count'].values)
    print(f"Exposure Inequality (Gini Coefficient): {gini:.4f} (0=perfect equality, 1=perfect inequality)")
    
    tier_diversity = 0.0
    if song_metadata is not None and 'artist_tier' in song_metadata.columns:
        artist_tiers = song_metadata[['artist_name', 'artist_tier']].drop_duplicates()
        logger.info("Found tier information for %d unique artists", len(artist_tiers))
        
        exposure_df = exposure_df.merge(
            artist_tiers,
            on='artist_name',
            how='left'
        )
        exposure_df['artist_tier'] = exposure_df['artist_tier'].fillna('Unknown')
        
        tier_exposure = exposure_df.groupby('artist_tier')['exposure_count'].agg(['sum', 'mean', 'count']).fillna(0)
        total_recs = exposure_df['exposure_count'].sum()
        
        print("\nExposure by Artist Tier:")
        print(tier_exposure)
        
        tier_percentages = tier_exposure['sum'] / total_recs * 100
        print("\nPercentage of Recommendations by Tier:")
        for tier, percentage in tier_percentages.items():
            print(f"• {tier}: {percentage:.1f}%")
        
        tier_diversity = 1 - (tier_exposure['sum'].max() / total_recs if total_recs > 0 else 0)
        print(f"Tier Diversity Score: {tier_diversity:.2f} (higher is more diverse)")
        
        top_artists = exposure_df.sort_values('exposure_count', ascending=False).head(5)
        bottom_artists = exposure_df[exposure_df['exposure_count'] > 0].sort_values('exposure_count').head(5)
        
        print("\nMost Exposed Artists:")
        for _, row in top_artists.iterrows():
            print(f"• {row['artist_name']}: {row['exposure_count']} recommendations (Tier: {row['artist_tier']})")
        
        print("\nLeast Exposed Artists:")
        for _, row in bottom_artists.iterrows():
            print(f"• {row['artist_name']}: {row['exposure_count']} recommendations (Tier: {row['artist_tier']})")
    
    logger.info("Exposure analysis complete")
    return {
        'exposure_df': exposure_df,
        'tier_diversity': tier_diversity,
        'gini_coefficient': gini
    }

def calculate_gini_coefficient(values: np.ndarray) -> float:
    """Calculate the Gini coefficient for a set of values.
    
    Args:
        values: Array of values
        
    Returns:
        Gini coefficient (0=perfect equality, 1=perfect inequality)
    """
    sorted_values = np.sort(values)
    n = len(sorted_values)
    
    if n == 0 or np.sum(sorted_values) == 0:
        return 0.0
    
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * sorted_values)) / (n * np.sum(sorted_values))