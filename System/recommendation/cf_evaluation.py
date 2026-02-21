from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

def evaluate_recommendations(
    recommendations: pd.DataFrame,
    test_interactions: pd.DataFrame,
    item_features: Optional[np.ndarray],
    item_ids: List[str],
    song_metadata: pd.DataFrame,
    k_values: List[int] = [5, 10]
) -> Dict[str, Dict[int, float]]:
    """Evaluate recommendation quality for collaborative filtering, computing Diversity if item_features is provided."""
    if not {'user_id', 'song_id', 'score', 'artist_tier'}.issubset(recommendations.columns):
        raise ValueError("Recommendations must contain 'user_id', 'song_id', 'score', 'artist_tier' columns")
    if not {'user_id', 'song_id'}.issubset(test_interactions.columns):
        raise ValueError("Test interactions must contain 'user_id', 'song_id' columns")

    metrics = {
        'Precision': {}, 'Genre Precision': {}, 'Language Precision': {},
        'Recall': {}, 'NDCG': {}, 'Hit Rate': {}, 'Emerging Artist Hit Rate': {},
        'Coverage (%)': {}, 'Diversity': {}, 'Novelty': {}, 'Emerging Artist Exposure Index': {}
    }
    
    # Initialize mappings for recommended songs metadata
    id_col = next((col for col in song_metadata.columns if col in ['song_id', 'track_id', 'id']), song_metadata.columns[0])
    song_to_genre = dict(zip(song_metadata[id_col], song_metadata.get('top_genre', pd.Series(index=song_metadata[id_col], dtype=str))))
    song_to_language = dict(zip(song_metadata[id_col], song_metadata.get('language', pd.Series(index=song_metadata[id_col], dtype=str))))
    song_to_tier = dict(zip(song_metadata[id_col], song_metadata.get('artist_tier', pd.Series(index=song_metadata[id_col], dtype=str))))
    
    # Group recommendations and test interactions by user
    recs_by_user = recommendations.groupby('user_id').apply(
        lambda x: x.sort_values('score', ascending=False)[['song_id', 'artist_tier']].head(max(k_values))
    ).reset_index()
    test_by_user = test_interactions.groupby('user_id')['song_id'].apply(set).to_dict()
    
    # --- REVISED MERGE AND COLUMN HANDLING FOR test_interactions_with_metadata ---
    # Merge test_interactions with ALL of song_metadata, then select/handle columns
    test_interactions_with_metadata = test_interactions.merge(
        song_metadata,
        left_on='song_id', right_on=id_col, how='left', suffixes=('', '_meta')
    )
    
    # Ensure 'top_genre' and 'language' columns exist and fill NaNs
    if 'top_genre' not in test_interactions_with_metadata.columns:
        test_interactions_with_metadata['top_genre'] = 'unknown_genre'
        logger.warning("Added missing 'top_genre' column to test_interactions_with_metadata with 'unknown_genre'.")
    else:
        test_interactions_with_metadata['top_genre'] = test_interactions_with_metadata['top_genre'].fillna('unknown_genre')

    if 'language' not in test_interactions_with_metadata.columns:
        test_interactions_with_metadata['language'] = 'unknown_language'
        logger.warning("Added missing 'language' column to test_interactions_with_metadata with 'unknown_language'.")
    else:
        test_interactions_with_metadata['language'] = test_interactions_with_metadata['language'].fillna('unknown_language')
    # --- END REVISED MERGE AND COLUMN HANDLING ---
    
    # Get user's relevant genres and languages from test interactions
    test_genres_by_user = test_interactions_with_metadata.groupby('user_id')['top_genre'].apply(lambda x: set(x.dropna())).to_dict()
    test_languages_by_user = test_interactions_with_metadata.groupby('user_id')['language'].apply(lambda x: set(x.dropna())).to_dict()
    
    # Calculate emerging artist proportions in catalog
    emerging_tiers = ['emerging_new', 'emerging_trending']
    total_songs = len(song_metadata)
    emerging_songs = song_metadata[song_metadata['artist_tier'].isin(emerging_tiers)]['song_id'].nunique() if 'artist_tier' in song_metadata.columns else 0
    catalog_emerging_proportion = emerging_songs / total_songs if total_songs > 0 else 0.0
    
    logger.info("Computing evaluation metrics for %d k-values", len(k_values))
    for k in k_values:
        precisions, genre_precisions, language_precisions = [], [], []
        recalls, ndcgs, hits, emerging_hits = [], [], [], []
        diversities, novelties = [], []
        emerging_recs_count = 0
        total_recs = 0
        unique_recs = set()
        
        for user_id in test_by_user:
            # Get top-k recommendations
            user_recs = recs_by_user[recs_by_user['user_id'] == user_id][['song_id', 'artist_tier']].head(k)
            if user_recs.empty:
                continue
                
            rec_songs = user_recs['song_id'].tolist()
            relevant_items = test_by_user.get(user_id, set())
            if not relevant_items:
                continue # Cannot calculate metrics if no relevant items in test set
                
            # Update coverage and emerging artist count
            unique_recs.update(rec_songs)
            total_recs += len(rec_songs)
            emerging_recs_count += sum(1 for _, row in user_recs.iterrows() if row['artist_tier'] in emerging_tiers)
            
            # Precision (exact song matches)
            recommended_set = set(rec_songs)
            hits_count = len(recommended_set & relevant_items)
            precision = hits_count / k if k > 0 else 0.0
            precisions.append(precision)
            
            # --- MODIFICATION FOR PER-SONG GENRE/LANGUAGE PRECISION ---
            user_relevant_genres_set = test_genres_by_user.get(user_id, set())
            user_relevant_languages_set = test_languages_by_user.get(user_id, set())

            genre_hits_per_song = 0
            for song_id in rec_songs:
                song_genre = song_to_genre.get(song_id, '')
                if song_genre and song_genre != 'unknown_genre' and song_genre in user_relevant_genres_set:
                    genre_hits_per_song += 1
            genre_precision = genre_hits_per_song / k if k > 0 else 0.0
            genre_precisions.append(genre_precision)
            
            language_hits_per_song = 0
            for song_id in rec_songs:
                song_language = song_to_language.get(song_id, '')
                if song_language and song_language != 'unknown_language' and song_language in user_relevant_languages_set:
                    language_hits_per_song += 1
            language_precision = language_hits_per_song / k if k > 0 else 0.0
            language_precisions.append(language_precision)
            # --- END OF MODIFICATION ---
            
            # Recall
            recall = hits_count / len(relevant_items) if len(relevant_items) > 0 else 0.0
            recalls.append(recall)
            
            # NDCG
            dcg = 0.0
            idcg = 0.0
            for i, item in enumerate(rec_songs[:k]):
                rel = 1.0 if item in relevant_items else 0.0
                dcg += rel / np.log2(i + 2)
                if i < len(relevant_items):
                    idcg += 1.0 / np.log2(i + 2)
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)
            
            # Hit Rate (exact song)
            hits.append(1.0 if hits_count > 0 else 0.0)
            
            # Emerging Artist Hit Rate
            has_emerging = any(row['artist_tier'] in emerging_tiers for _, row in user_recs.iterrows())
            emerging_hits.append(1.0 if has_emerging else 0.0)
            
            # Diversity
            if item_features is not None:
                rec_indices = [item_ids.index(item) for item in rec_songs if item in item_ids]
                if len(rec_indices) > 1:
                    rec_features = item_features[rec_indices]
                    sim_matrix = cosine_similarity(rec_features)
                    diversity = np.mean(1 - sim_matrix[np.triu_indices(len(rec_indices), k=1)])
                    diversities.append(diversity)
            
            # Novelty
            tier_weights = {
                'emerging_new': 1.0, 'emerging_trending': 0.8,
                'rising_established': 0.6, 'mid_tier': 0.4, 'established': 0.2,
                'established_trending': 0.2, 'established_legacy': 0.2
            }
            
            novelty_scores = [
                tier_weights.get(song_to_tier.get(item, 'established'), 0.2)
                for item in rec_songs
            ]
            novelties.append(np.mean(novelty_scores) if novelty_scores else 0.0)
        
        # Average metrics
        metrics['Precision'][k] = np.mean(precisions) if precisions else 0.0
        metrics['Genre Precision'][k] = np.mean(genre_precisions) if genre_precisions else 0.0
        metrics['Language Precision'][k] = np.mean(language_precisions) if language_precisions else 0.0
        metrics['Recall'][k] = np.mean(recalls) if recalls else 0.0
        metrics['NDCG'][k] = np.mean(ndcgs) if ndcgs else 0.0
        metrics['Hit Rate'][k] = np.mean(hits) if hits else 0.0
        metrics['Emerging Artist Hit Rate'][k] = np.mean(emerging_hits) if emerging_hits else 0.0
        metrics['Coverage (%)'][k] = len(unique_recs) / len(item_ids) * 100 if item_ids else 0.0
        metrics['Diversity'][k] = np.mean(diversities) if diversities else 0.0
        metrics['Novelty'][k] = np.mean(novelties) if novelties else 0.0
        metrics['Emerging Artist Exposure Index'][k] = (
            (emerging_recs_count / total_recs) / catalog_emerging_proportion
            if total_recs > 0 and catalog_emerging_proportion > 0 else 0.0
        )
    
    logger.info("Evaluation metrics computed successfully")
    return metrics

class CFRecommendationEvaluator:
    def __init__(
        self,
        song_features: Optional[pd.DataFrame],
        item_ids: List[str],
        song_metadata: pd.DataFrame,
        k_values: List[int] = [5, 10]
    ):
        """Initialize evaluator for collaborative filtering, building item_features from song_features."""
        self.item_ids = item_ids
        self.song_metadata = song_metadata
        self.k_values = k_values
        
        # Build item_features from song_features
        self.item_features = None
        if song_features is not None and not song_features.empty:
            try:
                # Ensure song_features is aligned with item_ids
                feature_cols = [col for col in song_features.columns if col != 'song_id']
                if not feature_cols:
                    logger.warning("No feature columns found in song_features; Diversity will be 0")
                else:
                    # Create item_features array in order of item_ids
                    song_features_indexed = song_features.set_index('song_id')
                    feature_matrix = []
                    for item_id in item_ids:
                        if item_id in song_features_indexed.index:
                            features = song_features_indexed.loc[item_id, feature_cols].values
                            feature_matrix.append(features)
                        else:
                            # Use zero vector for missing items
                            feature_matrix.append(np.zeros(len(feature_cols)))
                    self.item_features = np.array(feature_matrix)
                    logger.info("Built item_features for %d items from song_features", len(item_ids))
            except Exception as e:
                logger.error("Failed to build item_features from song_features: %s", str(e))
                self.item_features = None
        
        if not item_ids:
            raise ValueError("item_ids cannot be empty")
        if song_metadata.empty:
            logger.warning("Empty song_metadata provided; some metrics may be affected")
        if self.item_features is not None and (len(self.item_features.shape) < 1 or len(item_ids) != self.item_features.shape[0]):
            raise ValueError("item_ids must match item_features dimensions")

    def evaluate(
        self,
        recommendations: pd.DataFrame,
        test_interactions: pd.DataFrame
    ) -> Dict[str, Dict[int, float]]:
        return evaluate_recommendations(
            recommendations=recommendations,
            test_interactions=test_interactions,
            item_features=self.item_features,
            item_ids=self.item_ids,
            song_metadata=self.song_metadata,
            k_values=self.k_values
        )
