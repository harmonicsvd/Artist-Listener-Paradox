import pandas as pd
import numpy as np
import os
import sys
import logging
import pickle # For loading saved mappings
from collections import defaultdict
import random # Not directly used for sampling users anymore, but kept if other random ops exist
from tqdm.auto import tqdm # For progress bar
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity  # ADD THIS IMPORT

# Add project root to sys.path to import your existing modules
# Adjusted path: '..' takes it from 'recommendation' to 'System', another '..' takes it to 'Recommenation-System'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import your data loading and mapping utilities
from System.recommendation.recommendation_system import RecommendationSystem
from System.recommendation.utils.mappings import create_song_to_tier_mapping

# Re-import evaluators for metric computation
from System.recommendation.evaluation import RecommendationEvaluator
from System.recommendation.cf_evaluation import CFRecommendationEvaluator

# Set up basic logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Function to format and print metrics ---
def format_metrics(metrics: dict, k_values: list) -> None:
    """Format and print evaluation metrics."""
    print("\n" + "="*50)
    print("EVALUATION RESULTS (Individual Metrics)")
    print("="*50)
    header = f"{'Metric':<25} " + " ".join(f"k={k:<10}" for k in k_values)
    print(header)
    print("-" * (25 + len(k_values) * 11))
    for metric in [
        'Precision', 'Genre Precision', 'Language Precision',
        'Recall', 'NDCG', 'Hit Rate', 'Emerging Artist Hit Rate',
        'Coverage (%)', 'Diversity', 'Novelty', 'Emerging Artist Exposure Index'
    ]:
        values = metrics.get(metric, {})
        row = f"{metric:<25} " + " ".join(
            f"{values.get(k, 0.0):<10.4f}" if metric != 'Coverage (%)'
            else f"{values.get(k, 0.0):<10.2f}%" for k in k_values
        )
        print(row)
    print("="*50)
# --- End of function ---

class SimpleExposureTracker:
    """A simplified class to track artist exposure and calculate diversity/gini.
    This class is primarily used for the Gini and Tier Diversity metrics within the
    individual metric evaluation, not for a combined AS score."""
    
    def __init__(self):
        self.artist_exposure = defaultdict(int)
        self.recommendation_count = 0
        self.artist_recommendation_counts = defaultdict(int)
    
    def track_exposure(self, recommendations_df: pd.DataFrame, song_to_artist: dict):
        """Tracks artist exposure from a batch of recommendations."""
        if 'artist_id' in recommendations_df.columns:
            for artist_id in recommendations_df['artist_id'].dropna().unique():
                self.artist_exposure[artist_id] += 1
            for artist_id in recommendations_df['artist_id'].dropna():
                self.artist_recommendation_counts[artist_id] += 1
            self.recommendation_count += len(recommendations_df)
        else:
            logger.warning("No 'artist_id' column in recommendations for exposure tracking.")
    
    def analyze_exposure_distribution(self, song_metadata: pd.DataFrame):
        """Calculates simplified tier diversity and Gini coefficient."""
        if not self.artist_recommendation_counts:
            return {'tier_diversity': 0.0, 'gini_coefficient': 1.0} # Default to no diversity, max inequality
        
        # For Gini, we need the counts of recommendations per artist
        artist_recs_values = np.array(list(self.artist_recommendation_counts.values()))
        if len(artist_recs_values) == 0:
            gini_coefficient = 1.0
        else:
            # Calculate Gini coefficient (simplified)
            # Sort values
            sorted_recs = np.sort(artist_recs_values)
            n = len(sorted_recs)
            if n <= 1: # Gini is undefined or 0 for 0 or 1 item
                gini_coefficient = 0.0 if n == 1 else 1.0
            else:
                numerator = np.sum([(i + 1) * sorted_recs[i] for i in range(n)])
                denominator = n * np.sum(sorted_recs)
                if denominator == 0: # Avoid division by zero if all counts are zero
                    gini_coefficient = 1.0
                else:
                    gini_coefficient = (2 * numerator - (n + 1) * np.sum(sorted_recs)) / denominator
        
        # For Tier Diversity, we need artist tiers
        # This is a simplification, your actual system has detailed tier mapping
        tier_counts = defaultdict(int)
        total_artists_tracked = 0
        if 'artist_id' in song_metadata.columns and 'artist_tier' in song_metadata.columns:
            # Get tiers for artists that received recommendations
            artists_with_recs = list(self.artist_recommendation_counts.keys())
            artist_tier_map = song_metadata.set_index('artist_id')['artist_tier'].dropna().to_dict()
            for artist_id in artists_with_recs:
                tier = artist_tier_map.get(artist_id, 'unknown_tier')
                tier_counts[tier] += 1
                total_artists_tracked += 1
        
        tier_diversity = 0.0
        if total_artists_tracked > 0:
            # Simple diversity: proportion of unique tiers
            num_unique_tiers = len(tier_counts)
            if num_unique_tiers > 1:
                tier_diversity = num_unique_tiers / len(set(song_metadata['artist_tier'].dropna())) # Normalize by total possible tiers
            else:
                tier_diversity = 0.0 # No diversity if only one tier
        
        return {
            'tier_diversity': tier_diversity,
            'gini_coefficient': gini_coefficient
        }

class SimpleContentBasedRecommender:
    """A simple Content-Based Recommender with proper normalization applied to the features."""

    def __init__(self):
        self.item_features = None
        self.item_ids = []
        self.scaler = MinMaxScaler()  # Use MinMaxScaler as in advanced system; change to StandardScaler if needed
        self.song_metadata = None  # Store metadata for display
        self.song_id_to_index = {}  # For evaluators
        self.song_to_artist = {}  # For evaluators
        self.song_to_tier = {}  # For evaluators

    def train(self, song_metadata: pd.DataFrame, song_features: pd.DataFrame):
        """
        Trains the content-based recommender by processing song features.
        Uses your song_features DataFrame directly.
        """
        logger.info("Training Content-Based Recommender (baseline, with consistent normalization)...")
        self.song_metadata = song_metadata.copy()

        # Filter song_features to only include songs present in song_metadata
        common_song_ids = list(set(song_metadata['song_id']).intersection(set(song_features['song_id'])))
        filtered_song_features = song_features[song_features['song_id'].isin(common_song_ids)].copy()

        if filtered_song_features.empty:
            logger.error("No common songs between song_metadata and song_features for Content-Based training.")
            return

        self.item_ids = filtered_song_features['song_id'].tolist()
        self.song_id_to_index = {song_id: i for i, song_id in enumerate(self.item_ids)}

        # Populate song_to_artist and song_to_tier for evaluators
        self.song_to_artist = song_metadata.set_index('song_id')['artist_id'].dropna().to_dict()
        self.song_to_tier = song_metadata.set_index('song_id')['artist_tier'].dropna().to_dict()

        # Select only numerical columns (not song_id)
        feature_columns = [col for col in filtered_song_features.columns if col != 'song_id' and pd.api.types.is_numeric_dtype(filtered_song_features[col])]
        if not feature_columns:
            logger.error("No numerical features found in song_features for Content-Based training. Please check your data.")
            return

        # 1. Extract raw features (as float)
        self.item_features = filtered_song_features[feature_columns].astype(float).values

        # 2. Fill NaNs or infinities
        self.item_features = np.nan_to_num(self.item_features, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. Normalization: Use same scaler as advanced system (MinMaxScaler/StandardScaler, change if needed)
        self.item_features = self.scaler.fit_transform(self.item_features)

        logger.info(f"Content-Based Recommender trained with {len(self.item_ids)} items and {self.item_features.shape[1]} normalized features.")

    def recommend(self, user_id: str, user_interactions: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """
        Generates recommendations for a user based on their liked songs' normalized features.
        User profile is a simple average of features of liked songs.
        """
        if self.item_features is None:
            logger.error("Content-Based Recommender not trained. Call .train() first.")
            return pd.DataFrame(columns=['user_id', 'song_id', 'score'])

        user_liked_songs = user_interactions[user_interactions['user_id'] == user_id]['song_id'].tolist()

        if not user_liked_songs:
            logger.warning(f"User {user_id} has no liked songs for content-based recommendation. Returning random.")
            return self._get_random_recommendations(user_id, n)

        user_profile_features = []
        for song_id in user_liked_songs:
            idx = self.song_id_to_index.get(song_id)
            if idx is not None:
                user_profile_features.append(self.item_features[idx])
            else:
                logger.debug(f"Song {song_id} not found in trained items; skipping.")

        if not user_profile_features:
            logger.warning(f"No features found for user {user_id}'s liked songs. Returning empty.")
            return pd.DataFrame(columns=['user_id', 'song_id', 'score'])

        user_profile = np.mean(user_profile_features, axis=0).reshape(1, -1)
        # Cosine similarity on normalized features
        scores = cosine_similarity(user_profile, self.item_features).flatten()

        recommendations_df = pd.DataFrame({
            'song_id': self.item_ids,
            'score': scores
        })

        # Exclude already liked songs
        recommendations_df = recommendations_df[~recommendations_df['song_id'].isin(user_liked_songs)]
        recommendations_df = recommendations_df.sort_values(by='score', ascending=False).head(n)
        recommendations_df['user_id'] = user_id
        return recommendations_df

    def _get_random_recommendations(self, user_id: str, n: int) -> pd.DataFrame:
        """Helper to return random recommendations."""
        if not self.item_ids:
            return pd.DataFrame(columns=['user_id', 'song_id', 'score'])
        random_songs = np.random.choice(self.item_ids, min(n, len(self.item_ids)), replace=False)
        return pd.DataFrame({'user_id': user_id, 'song_id': random_songs, 'score': np.random.rand(len(random_songs))})


class SimpleMatrixFactorizationRecommender:
    """A very basic Matrix Factorization Recommender using SVD."""
    
    def __init__(self, n_factors: int = 200):  # UPDATED: Changed from 5 to 200
        self.n_factors = n_factors
        self.user_item_matrix = None
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.U = None # User latent factors
        self.Vt = None # Item latent factors (transposed)
        self.song_metadata = None # Store metadata for display
        self.song_id_to_index = {} # For evaluators
        self.song_to_artist = {} # For evaluators
        self.song_to_tier = {} # For evaluators
    
    def train(self, user_interactions: pd.DataFrame, song_metadata: pd.DataFrame):
        """
        Trains the MF recommender by performing SVD on the user-item interaction matrix.
        Assumes implicit feedback (interaction = 1).
        """
        logger.info("Training Simple Matrix Factorization Recommender...")
        self.song_metadata = song_metadata.copy() # Keep a copy of metadata
        
        # Create mappings for users and items
        all_users = user_interactions['user_id'].unique()
        all_items = user_interactions['song_id'].unique()
        
        self.user_map = {user: i for i, user in enumerate(all_users)}
        self.item_map = {item: i for i, item in enumerate(all_items)}
        self.reverse_user_map = {i: user for user, i in self.user_map.items()}
        self.reverse_item_map = {i: item for item, i in self.item_map.items()}
        self.song_id_to_index = self.item_map # MF's item_map is its song_id_to_index
        
        # Populate song_to_artist and song_to_tier for evaluators
        self.song_to_artist = song_metadata.set_index('song_id')['artist_id'].dropna().to_dict()
        self.song_to_tier = song_metadata.set_index('song_id')['artist_tier'].dropna().to_dict()
        
        # Create user-item matrix
        num_users = len(all_users)
        num_items = len(all_items)
        self.user_item_matrix = np.zeros((num_users, num_items))
        
        for _, row in user_interactions.iterrows():
            user_idx = self.user_map.get(row['user_id'])
            item_idx = self.item_map.get(row['song_id'])
            if user_idx is not None and item_idx is not None:
                self.user_item_matrix[user_idx, item_idx] = 1 # Implicit feedback
        
        # Perform SVD
        # svds returns U, s, Vt. We need U and Vt for predictions.
        # Ensure n_factors is less than min(num_users, num_items)
        k_svd = min(self.n_factors, num_users - 1, num_items - 1)
        if k_svd <= 0:
            logger.warning("Not enough users or items for SVD with n_factors. Skipping SVD.")
            self.U = None
            self.Vt = None
            return
        
        try:
            self.U, s, self.Vt = svds(self.user_item_matrix, k=k_svd)
            # Convert singular values to a diagonal matrix
            s_diag_matrix = np.diag(s)
            # Reconstruct the original matrix for prediction: U * S * Vt
            self.U = np.dot(self.U, s_diag_matrix)
            logger.info(f"MF Recommender trained with {k_svd} factors.")
        except Exception as e:
            logger.error(f"Error during SVD training: {e}. MF recommender might not work.")
            self.U = None
            self.Vt = None
    
    def recommend(self, user_id: str, user_interactions: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """
        Generates recommendations for a user by predicting ratings using MF.
        """
        if self.U is None or self.Vt is None:
            logger.error("MF Recommender not trained or SVD failed. Returning empty recommendations.")
            return pd.DataFrame(columns=['user_id', 'song_id', 'score'])
        
        if user_id not in self.user_map:
            logger.warning(f"User {user_id} not found in MF model. Cannot make recommendations.")
            return pd.DataFrame(columns=['user_id', 'song_id', 'score'])
        
        user_idx = self.user_map[user_id]
        
        # Predict ratings for all items for the given user
        # Predicted ratings = U_user_row * Vt
        predicted_ratings = np.dot(self.U[user_idx, :], self.Vt)
        
        # Create a DataFrame of scores
        all_item_ids = [self.reverse_item_map[i] for i in range(len(self.item_map))]
        recommendations_df = pd.DataFrame({
            'song_id': all_item_ids,
            'score': predicted_ratings
        })
        
        # Exclude songs already liked by the user
        user_liked_songs = user_interactions[user_interactions['user_id'] == user_id]['song_id'].tolist()
        recommendations_df = recommendations_df[~recommendations_df['song_id'].isin(user_liked_songs)]
        
        # Sort by score and take top N
        recommendations_df = recommendations_df.sort_values(by='score', ascending=False).head(n)
        recommendations_df['user_id'] = user_id # Add user_id for consistency
        
        return recommendations_df

class SimpleHybridRecommender:
    """
    A very basic Hybrid Recommender combining Content-Based and Matrix Factorization.
    It takes pre-trained instances of both recommenders.
    """
    
    def __init__(self, cb_recommender: SimpleContentBasedRecommender, mf_recommender: SimpleMatrixFactorizationRecommender, content_weight: float = 0.5, mf_weight: float = 0.5):
        self.cb_recommender = cb_recommender
        self.mf_recommender = mf_recommender
        self.content_weight = content_weight
        self.mf_weight = mf_weight
        
        # Hybrid recommender needs to expose metadata/mappings from its components for evaluation
        self.song_metadata = self.cb_recommender.song_metadata # Or mf_recommender.song_metadata
        self.song_features = self.cb_recommender.item_features # From CB
        self.item_ids = self.cb_recommender.item_ids # From CB
        self.song_id_to_index = self.cb_recommender.song_id_to_index # From CB
        self.song_to_artist = self.cb_recommender.song_to_artist # From CB
        self.song_to_tier = self.cb_recommender.song_to_tier # From CB
    
    def train(self, *args, **kwargs):
        """Hybrid recommender doesn't train itself, relies on components being trained."""
        logger.info("Simple Hybrid Recommender: Components are assumed to be trained.")
    
    def recommend(self, user_id: str, user_interactions: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """
        Generates hybrid recommendations by combining scores from CB and MF.
        Scores are normalized before combining.
        """
        # Get recommendations from Content-Based Recommender
        # Request scores for all items to ensure comprehensive merging
        cb_recs = self.cb_recommender.recommend(user_id, user_interactions, n=len(self.cb_recommender.item_ids))
        if cb_recs.empty:
            logger.warning("Content-Based recommender returned no recommendations.")
            return pd.DataFrame(columns=['user_id', 'song_id', 'score'])
        
        # Get recommendations from Matrix Factorization Recommender
        # Request scores for all items to ensure comprehensive merging
        mf_recs = self.mf_recommender.recommend(user_id, user_interactions, n=len(self.mf_recommender.item_map))
        if mf_recs.empty:
            logger.warning("Matrix Factorization recommender returned no recommendations.")
            return pd.DataFrame(columns=['user_id', 'song_id', 'score'])
        
        # Normalize scores to a 0-1 range before combining
        scaler = MinMaxScaler()
        # Ensure there are values to scale
        if not cb_recs['score'].empty:
            cb_recs['score_normalized'] = scaler.fit_transform(cb_recs[['score']])
        else:
            cb_recs['score_normalized'] = 0.0 # Default if no scores
        
        if not mf_recs['score'].empty:
            mf_recs['score_normalized'] = scaler.fit_transform(mf_recs[['score']])
        else:
            mf_recs['score_normalized'] = 0.0 # Default if no scores
        
        # Merge the scores from both recommenders
        combined_recs = pd.merge(cb_recs[['song_id', 'score_normalized']].rename(columns={'score_normalized': 'cb_score'}),
                                mf_recs[['song_id', 'score_normalized']].rename(columns={'score_normalized': 'mf_score'}),
                                on='song_id',
                                how='outer').fillna(0) # Fill missing scores with 0
        
        # Combine scores using weighted sum
        combined_recs['score'] = (self.content_weight * combined_recs['cb_score'] +
                                 self.mf_weight * combined_recs['mf_score'])
        
        # Exclude songs already liked by the user (re-apply in case outer join brought them back)
        user_liked_songs = user_interactions[user_interactions['user_id'] == user_id]['song_id'].tolist()
        final_recs = combined_recs[~combined_recs['song_id'].isin(user_liked_songs)]
        
        # Sort by score and take top N
        final_recs = final_recs.sort_values(by='score', ascending=False).head(n)
        final_recs['user_id'] = user_id # Add user_id for consistency
        
        return final_recs
# --- Data Loading and Preprocessing (Adapted from your run_system4.py) ---

def load_and_prepare_data(config):
    """Loads and prepares data similar to your main script's initialize_shared_resources."""
    logger.info("Loading and preparing data using your project's utilities...")
    
    system = RecommendationSystem(data_dir=config['data_path'])
    system.load_data(max_users=config['max_users'])
    
    user_interactions_raw = system.data_manager.user_loader.interactions
    song_metadata_raw = system.data_manager.song_loader.song_metadata
    song_features_raw = system.data_manager.song_loader.song_features
    artist_identification_raw = system.data_manager.artist_identification
    
    if song_metadata_raw.empty or user_interactions_raw.empty or song_features_raw.empty:
        logger.error("No data loaded from initial DataManager setup. Please check data paths.")
        raise ValueError("Failed to load data for simple system comparison.")
    
    # Simple consistency check for this script:
    metadata_songs = set(song_metadata_raw['song_id'])
    feature_songs = set(song_features_raw['song_id'])
    consistent_item_ids = metadata_songs.intersection(feature_songs)
    
    filtered_song_metadata = song_metadata_raw[song_metadata_raw['song_id'].isin(consistent_item_ids)].copy()
    filtered_song_features = song_features_raw[song_features_raw['song_id'].isin(consistent_item_ids)].copy()
    filtered_interactions = user_interactions_raw[user_interactions_raw['song_id'].isin(consistent_item_ids)].copy()
    
    logger.info(f"Loaded {len(filtered_interactions)} interactions, {len(filtered_song_metadata)} songs metadata, {len(filtered_song_features)} song features.")
    
    # Ensure 'genre', 'top_genre', 'language', and 'artist_id' columns are present in filtered_song_metadata
    # This is crucial for downstream components (recommenders, tier analysis)
    if 'top_genre' in filtered_song_metadata.columns:
        filtered_song_metadata['genre'] = filtered_song_metadata['top_genre']
    elif 'genre' not in filtered_song_metadata.columns:
        filtered_song_metadata['genre'] = 'unknown_genre' # Fallback if neither top_genre nor genre exists
    
    if 'top_genre' not in filtered_song_metadata.columns: # Ensure top_genre exists, even if derived from 'genre'
        filtered_song_metadata['top_genre'] = filtered_song_metadata['genre'] # Use 'genre' as 'top_genre' if original was missing
        logger.warning("Added missing 'top_genre' column to filtered_song_metadata from 'genre'.")
    
    filtered_song_metadata['top_genre'] = filtered_song_metadata['top_genre'].fillna('unknown_genre')
    
    if 'language' not in filtered_song_metadata.columns:
        filtered_song_metadata['language'] = 'unknown_language'
        logger.warning("Added missing 'language' column to filtered_song_metadata with 'unknown_language'.")
    
    filtered_song_metadata['language'] = filtered_song_metadata['language'].fillna('unknown_language') # Fill any NaNs
    
    if 'artist_id' not in filtered_song_metadata.columns:
        logger.warning("artist_id column missing in song_metadata. Attempting to derive from artist_name.")
        filtered_song_metadata['artist_id'] = filtered_song_metadata['artist_name'].fillna('unknown_artist_id')
    
    filtered_song_metadata['artist_id'] = filtered_song_metadata['artist_id'].fillna('unknown_artist_id') # Fill any NaNs
    
    # Use the proper mapping function from your utilities for artist_tier
    song_to_tier_map = create_song_to_tier_mapping(filtered_song_metadata)
    filtered_song_metadata['artist_tier'] = filtered_song_metadata['song_id'].map(song_to_tier_map).fillna('unknown_tier')
    logger.info("Artist tiers populated using create_song_to_tier_mapping.")
    
    return filtered_interactions, filtered_song_metadata, filtered_song_features

def split_data_for_simple_eval(interactions: pd.DataFrame, test_size=0.2, min_interactions=5):
    """
    Simple data split for evaluation: Ensures users have enough interactions for train/test.
    """
    user_counts = interactions['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    filtered_interactions = interactions[interactions['user_id'].isin(valid_users)].copy()
    
    train_interactions = []
    test_interactions = []
    
    for user_id in valid_users:
        user_data = filtered_interactions[filtered_interactions['user_id'] == user_id]
        if len(user_data) >= min_interactions:
            train, test = train_test_split(user_data, test_size=test_size, random_state=42)
            train_interactions.append(train)
            test_interactions.append(test)
    
    if not train_interactions or not test_interactions:
        logger.warning("Not enough users or interactions for train/test split. Returning empty DataFrames.")
        return pd.DataFrame(), pd.DataFrame(), []
    
    train_df = pd.concat(train_interactions, ignore_index=True)
    test_df = pd.concat(test_interactions, ignore_index=True)
    eval_users = train_df['user_id'].unique().tolist() # Users present in train and potentially test
    
    logger.info(f"Data split for evaluation: Train {len(train_df)} interactions, Test {len(test_df)} interactions for {len(eval_users)} users.")
    return train_df, test_df, eval_users

def perform_individual_metrics_evaluation(recommender, recommender_name: str, train_interactions: pd.DataFrame, test_interactions: pd.DataFrame, eval_users_subset: list, song_metadata: pd.DataFrame, song_features: pd.DataFrame, k_values: list):
    """
    Performs evaluation for a given simple recommender across a subset of eval_users and k_values,
    calculating individual LS and AS metrics without combining them into a single loss.
    """
    logger.info(f"\n--- Starting Individual Metrics Evaluation for {recommender_name} (on {len(eval_users_subset)} users) ---")
    
    metrics_all_k = {}
    for k in k_values:
        logger.info(f"Evaluating {recommender_name} at k={k} for {len(eval_users_subset)} users...")
        all_recs_for_k = []
        
        # Use tqdm for progress bar over user recommendations
        for user_id in tqdm(eval_users_subset, desc=f"Generating recs for {recommender_name} k={k}"):
            recs = recommender.recommend(user_id, train_interactions, n=k) # Train interactions for generating recs
            if not recs.empty:
                all_recs_for_k.append(recs)
        
        if not all_recs_for_k:
            logger.warning(f"No recommendations generated for {recommender_name} at k={k}. Skipping metrics for this k.")
            # Populate metrics with 0.0 if no recommendations
            for metric_name in ['Precision', 'Genre Precision', 'Language Precision', 'Recall', 'NDCG', 'Hit Rate', 'Emerging Artist Hit Rate', 'Coverage (%)', 'Diversity', 'Novelty', 'Emerging Artist Exposure Index', 'Tier Diversity', 'Gini Coefficient']:
                if metric_name not in metrics_all_k:
                    metrics_all_k[metric_name] = {}
                metrics_all_k[metric_name][k] = 0.0
            continue
        
        combined_recs_df = pd.concat(all_recs_for_k, ignore_index=True)
        
        # Ensure artist_id and artist_tier are available for evaluation metrics
        if 'artist_id' not in combined_recs_df.columns or 'artist_tier' not in combined_recs_df.columns:
            combined_recs_df = pd.merge(combined_recs_df, song_metadata[['song_id', 'artist_id', 'artist_tier']], on='song_id', how='left')
            combined_recs_df['artist_id'] = combined_recs_df['artist_id'].fillna('unknown_artist_id')
            combined_recs_df['artist_tier'] = combined_recs_df['artist_tier'].fillna('unknown_tier')
        
        # Initialize appropriate evaluator
        if isinstance(recommender, SimpleContentBasedRecommender):
            evaluator = RecommendationEvaluator(
                item_features=recommender.item_features,
                item_ids=recommender.item_ids,
                song_metadata=song_metadata, # Pass full song_metadata
                k_values=[k]
            )
        elif isinstance(recommender, (SimpleMatrixFactorizationRecommender, SimpleHybridRecommender)):
            # For CF and Hybrid, item_features is optional for Diversity calculation in CFRecommendationEvaluator
            # We pass song_features to allow it to build item_features if needed for Diversity.
            evaluator = CFRecommendationEvaluator(
                song_features=song_features, # Pass full song_features
                item_ids=list(recommender.song_id_to_index.keys()), # Use recommender's item_ids/mapping
                song_metadata=song_metadata, # Pass full song_metadata
                k_values=[k]
            )
        else:
            logger.error(f"Unsupported recommender type for evaluation: {type(recommender)}")
            continue
        
        # Evaluate metrics
        current_k_metrics = evaluator.evaluate(recommendations=combined_recs_df, test_interactions=test_interactions)
        for metric_name, values in current_k_metrics.items():
            if metric_name not in metrics_all_k:
                metrics_all_k[metric_name] = {}
            metrics_all_k[metric_name][k] = values.get(k, 0.0)
        
        # Calculate Tier Diversity and Gini Coefficient separately using SimpleExposureTracker
        # These are individual metrics, not part of a combined AS score for the baseline.
        temp_exposure_tracker = SimpleExposureTracker()
        temp_exposure_tracker.track_exposure(combined_recs_df, recommender.song_to_artist)
        exposure_result = temp_exposure_tracker.analyze_exposure_distribution(song_metadata=song_metadata)
        
        if 'Tier Diversity' not in metrics_all_k: metrics_all_k['Tier Diversity'] = {}
        metrics_all_k['Tier Diversity'][k] = exposure_result.get('tier_diversity', 0.0)
        
        if 'Gini Coefficient' not in metrics_all_k: metrics_all_k['Gini Coefficient'] = {}
        metrics_all_k['Gini Coefficient'][k] = exposure_result.get('gini_coefficient', 1.0)
    
    logger.info(f"--- Individual Metrics Evaluation for {recommender_name} Complete ---")
    return metrics_all_k

def print_tier_recommendation_statistics(recommender, recommender_name: str, sample_user_ids: list, train_interactions: pd.DataFrame, song_metadata: pd.DataFrame, n_recs: int):
    """
    Generates recommendations for a sample of users, aggregates them,
    and prints the absolute counts and percentages of recommendations per artist tier.
    """
    logger.info(f"\n--- Tier-wise Recommendation Statistics for {recommender_name} (k={n_recs}, {len(sample_user_ids)} Users) ---")
    
    all_sample_recs = []
    for user_id in tqdm(sample_user_ids, desc=f"Collecting recs for tier analysis ({recommender_name})"):
        recs = recommender.recommend(user_id, train_interactions, n=n_recs)
        if not recs.empty:
            all_sample_recs.append(recs)
    
    if not all_sample_recs:
        logger.warning(f"No recommendations generated for {recommender_name} in sample analysis.")
        print("-" * 60)
        print("No recommendations to analyze tier distribution.")
        print("-" * 60)
        return
    
    combined_sample_recs_df = pd.concat(all_sample_recs, ignore_index=True)
    
    # Merge artist_tier information
    combined_sample_recs_df = pd.merge(combined_sample_recs_df, song_metadata[['song_id', 'artist_tier']], on='song_id', how='left')
    combined_sample_recs_df['artist_tier'] = combined_sample_recs_df['artist_tier'].fillna('unknown_tier')
    
    # Calculate tier-wise counts and percentages
    tier_counts = combined_sample_recs_df['artist_tier'].value_counts()
    total_recs = tier_counts.sum()
    
    if total_recs == 0:
        logger.warning("No recommendations with tier information for percentage calculation.")
        print("-" * 60)
        print("No recommendations with tier information to analyze.")
        print("-" * 60)
        return
    
    tier_percentages = (tier_counts / total_recs) * 100
    
    print(f"\nRecommendation Distribution by Artist Tier for {recommender_name}:")
    print("-" * 60)
    print(f"{'Tier':<25} {'Count':<10} {'Percentage':<10}")
    print("-" * 60)
    for tier in tier_percentages.index:
        count = tier_counts[tier]
        percentage = tier_percentages[tier]
        print(f"{tier:<25} {count:<10} {percentage:.2f}%")
    print("-" * 60)
    print(f"{'Total Recommendations':<25} {total_recs:<10} {'100.00%':<10}")
    print("-" * 60)

# --- Main Execution ---
if __name__ == "__main__":
    # Define a simplified config based on your run_system4.py's base_config
    # Adjust paths as necessary for your environment
    config = {
        'data_path': '/Users/varadkulkarni/Thesis-FaRM/Recommenation-System/System/Finaldata', # Your data path
        'max_users': 40000, # Max users to load, consistent with your main script
        'n_recs': 5, # Number of recommendations to generate per user for analysis (for tier stats)
        'k_values_for_eval': [5], # K values for individual evaluation metrics
        'min_interactions_for_eval_split': 6, # Minimum interactions for a user to be included in eval split
    }
    
    # 1. Load and Prepare Data
    try:
        full_interactions, full_song_metadata, full_song_features = load_and_prepare_data(config)
    except ValueError as e:
        logger.critical(f"Exiting due to data loading error: {e}")
        sys.exit(1)
    
    if full_interactions.empty or full_song_metadata.empty or full_song_features.empty:
        logger.critical("Loaded data is empty. Cannot proceed with recommender training.")
        sys.exit(1)
    
    # 2. Split Data for Evaluation (Still needed to get train_interactions and eval_users)
    train_interactions_eval, test_interactions_eval, eval_users = split_data_for_simple_eval(
        full_interactions,
        test_size=0.5, # Using 0.5 test size as in your main script
        min_interactions=config['min_interactions_for_eval_split']
    )
    
    if not eval_users:
        logger.critical("No valid evaluation users after splitting. Cannot perform analysis.")
        sys.exit(1)
    
    # Use all eligible users for analysis and evaluation
    logger.info(f"Evaluating and analyzing for all {len(eval_users)} eligible users.")
    
    # 3. Initialize and Train Simple Recommenders
    logger.info("\n--- Initializing and Training Simple Recommenders ---")
    cb_recommender = SimpleContentBasedRecommender()
    cb_recommender.train(full_song_metadata, full_song_features)
    
    # UPDATED: Changed from n_factors=50 to n_factors=200
    mf_recommender = SimpleMatrixFactorizationRecommender(n_factors=100) 
    mf_recommender.train(train_interactions_eval, full_song_metadata) # MF needs training interactions
    
    hybrid_recommender = SimpleHybridRecommender(cb_recommender, mf_recommender, content_weight=0.5, mf_weight=0.5)
    hybrid_recommender.train() # No actual training for hybrid, just confirms components are ready
    
    # 4. Perform Individual Metrics Evaluation for Each Recommender
    all_recommenders_for_eval = {
        "Simple ContentBased": cb_recommender,
        "Simple MatrixFactorization": mf_recommender,
        "Simple HybridContentMF": hybrid_recommender
    }
    
    results_metrics = {}
    for name, recommender_obj in all_recommenders_for_eval.items():
        metrics = perform_individual_metrics_evaluation( # Call the new function
            recommender_obj,
            name,
            train_interactions_eval,
            test_interactions_eval,
            eval_users, # Use eval_users directly
            full_song_metadata,
            full_song_features,
            config['k_values_for_eval']
        )
        results_metrics[name] = metrics
    
    # 5. Display Individual Evaluation Results
    logger.info("\n" + "="*80)
    logger.info(f"INDIVIDUAL BASELINE METRICS (Averaged over {len(eval_users)} evaluation users)")
    logger.info("="*80)
    
    sorted_k_values = sorted(config['k_values_for_eval'])
    for name, metrics in results_metrics.items():
        print(f"\nResults for {name.upper()}")
        format_metrics(metrics, sorted_k_values)
    
    logger.info("="*80)
    logger.info("INDIVIDUAL BASELINE METRICS EVALUATION COMPLETE.")
    
    # 6. Tier-wise Artist Recommendation Statistics for All Eligible Users
    logger.info(f"\n--- Generating Tier-wise Artist Recommendation Statistics (Based on All {len(eval_users)} Eligible Users) ---")
    for name, recommender_obj in all_recommenders_for_eval.items():
        print_tier_recommendation_statistics(
            recommender_obj,
            name,
            eval_users, # Use eval_users directly
            train_interactions_eval, # Recommendations are based on training interactions
            full_song_metadata,
            config['n_recs'] # Use n_recs for the number of recommendations per user
        )
    
    logger.info("--- Baseline Tier-wise Analysis Complete ---")