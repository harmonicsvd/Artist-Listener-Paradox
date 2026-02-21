from System.recommendation.utils.similarity import compute_content_similarity
from typing import Dict, List, Optional, Set
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from System.recommendation.base import RecommenderBase
from System.recommendation.utils.mappings import create_song_to_tier_mapping, create_artist_metric_mappings, create_genre_mapping, create_language_mapping, compute_related_genres
from System.recommendation.utils.weights import calculate_language_weights, calculate_genre_weights, calculate_item_weights
import logging
from tqdm import tqdm
import os
import pickle

logger = logging.getLogger(__name__)

class ContentBasedRecommender(RecommenderBase):
    """Content-based recommender using item features and user profiles."""
    def __init__(
        self,
        tier_weights: Optional[Dict[str, float]] = None,
        excluded_features: Optional[List[str]] = None,
        feature_weights: Optional[Dict[str, float]] = None,
        user_profiles: Optional[Dict[str, Dict]] = None,
        user_extended_genres_cache: Optional[Dict[str, Set[str]]] = None,
        related_genres_map: Optional[Dict[str, Set[str]]] = None,
        song_to_tier: Optional[Dict[str, str]] = None,
        song_to_pop: Optional[Dict[str, float]] = None,
        song_to_fam: Optional[Dict[str, float]] = None,
        song_to_genres: Optional[Dict[str, List[str]]] = None,
        song_to_language: Optional[Dict[str, str]] = None,
        song_to_artist: Optional[Dict[str, str]] = None,
        name: str = "ContentBased"
    ):
        """Initialize the content-based recommender."""
        super().__init__(name=name)
        self.tier_weights = tier_weights
        self.excluded_features = excluded_features or []
        self.feature_weights = feature_weights or {
            'audio_features': 2.5,
            'genres': 15.0,
            'language': 4.0
        }
        self.user_profiles = user_profiles if user_profiles is not None else {}
        self.user_extended_genres_cache = user_extended_genres_cache if user_extended_genres_cache is not None else {}
        self.related_genres_map = related_genres_map
        self.song_to_tier = song_to_tier
        self.song_to_pop = song_to_pop
        self.song_to_fam = song_to_fam
        self.song_to_genres = song_to_genres
        self.song_to_language = song_to_language
        self.song_to_artist = song_to_artist
        self.item_features = None
        self.item_ids = None
        self.song_id_to_index = {}
        self.scaler = None
        self.song_metadata = None
        self.artist_identification = None

    def train(
        self,
        user_interactions: pd.DataFrame,
        song_features: pd.DataFrame,
        song_metadata: pd.DataFrame,
        artist_identification: Optional[pd.DataFrame] = None,
        testing_mode: bool = False,
        cache_dir: Optional[str] = None
    ) -> None:
        if testing_mode:
            logger.info("Training ContentBasedRecommender")

        super().train(user_interactions)
        self.song_metadata = song_metadata
        self.artist_identification = artist_identification

        if self.song_to_artist is None and 'artist_name' in song_metadata.columns:
            self.song_to_artist = dict(zip(song_metadata['song_id'], song_metadata['artist_name']))

        num_users = user_interactions['user_id'].nunique()
        num_songs = song_features['song_id'].nunique()
        cache_features_path = os.path.join(cache_dir, f"{self.name.lower()}_item_features_u{num_users}_s{num_songs}.npy") if cache_dir else None
        cache_similarity_path = os.path.join(cache_dir, f"{self.name.lower()}_similarity_matrix_u{num_users}_s{num_songs}.npy") if cache_dir else None
        cache_item_ids_path = os.path.join(cache_dir, f"{self.name.lower()}_item_ids_u{num_users}_s{num_songs}.pkl") if cache_dir else None
        cache_user_profiles_path = os.path.join(cache_dir, f"{self.name.lower()}_user_profiles_u{num_users}.pkl") if cache_dir else None

        if cache_dir and all(os.path.exists(p) for p in [cache_features_path, cache_similarity_path, cache_item_ids_path, cache_user_profiles_path]):
            logger.info(f"Attempting to load ContentBased matrices and profiles from cache: {num_users} users, {num_songs} songs.")
            try:
                self.item_features = np.load(cache_features_path)
                self.similarity_matrix = np.load(cache_similarity_path)
                self.item_ids = pd.read_pickle(cache_item_ids_path)
                self.song_id_to_index = {song_id: i for i, song_id in enumerate(self.item_ids)}
                with open(cache_user_profiles_path, 'rb') as f:
                    self.user_profiles = pickle.load(f)
                self.is_trained = True
                if testing_mode:
                    logger.info("ContentBased training complete (from cache).")
                return
            except Exception as e:
                logger.warning(f"Error loading ContentBased data from cache: {e}. Falling back to recalculation...")

        logger.info(f"Cache not found or failed to load for ContentBased data. Calculating from scratch...")
        id_col = next((col for col in song_features.columns if col in ['song_id', 'track_id', 'id']), song_features.columns[0])
        non_feature_cols_dynamic = ['song_id', 'artist_id', 'album_id', 'track_id'] + self.excluded_features
        feature_cols = [col for col in song_features.columns if col not in non_feature_cols_dynamic and song_features[col].dtype in ['int64', 'float64'] and not col.startswith('genre_')]

        if not feature_cols:
            logger.error("No valid numeric feature columns found for ContentBasedRecommender after exclusion. Check song_features.columns and self.excluded_features.")
            raise ValueError("No valid feature columns found for content-based recommendation.")

        self.item_features_names = feature_cols
        item_features_df = song_features.set_index(id_col)[feature_cols].copy()
        self.item_ids = item_features_df.index.tolist()
        self.item_features = item_features_df.fillna(0).to_numpy()

        logger.info("Normalizing item features...")
        self.scaler = StandardScaler()
        self.item_features = self.scaler.fit_transform(self.item_features)
        logger.info(f"Normalized {self.item_features.shape[0]} items with {self.item_features.shape[1]} features.")

        self.song_id_to_index = {song_id: i for i, song_id in enumerate(self.item_ids)}
        logger.info("Computing item similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.item_features)
        logger.info(f"Computed similarity matrix of shape: {self.similarity_matrix.shape}")

        logger.info("Building user profiles...")
        self.user_profiles = self._build_user_profiles(user_interactions, self.item_features, self.item_ids)
        logger.info(f"{self.name} recommender trained. Built {len(self.user_profiles)} user profiles.")

        self.is_trained = True

        if cache_dir:
            logger.info(f"Saving ContentBased matrices and profiles to cache: {num_users} users, {num_songs} songs.")
            np.save(cache_features_path, self.item_features)
            np.save(cache_similarity_path, self.similarity_matrix)
            pd.to_pickle(self.item_ids, cache_item_ids_path)
            with open(cache_user_profiles_path, 'wb') as f:
                pickle.dump(self.user_profiles, f)
            logger.info("ContentBased data saved to cache.")

    def _build_user_profiles(self, user_interactions: pd.DataFrame, scaled_item_features: np.ndarray, item_ids_list: List[str]) -> Dict[str, np.ndarray]:
        user_profiles = {}
        song_id_to_index_map = {song_id: i for i, song_id in enumerate(item_ids_list)}
        logger.debug(f"Building user profiles for {len(user_interactions['user_id'].unique())} unique users.")
        
        user_interactions_grouped = user_interactions.groupby('user_id')
        for user_id, group in tqdm(user_interactions_grouped, desc="Building user profiles", leave=False, disable=False):
            user_listened_songs = group['song_id'].tolist()
            listened_song_features = []
            for song_id in user_listened_songs:
                if song_id in song_id_to_index_map:
                    idx = song_id_to_index_map[song_id]
                    listened_song_features.append(scaled_item_features[idx])
                else:
                    logger.debug(f"User {user_id}: Song {song_id} from listened history not found in item_ids. Skipping features for this song.")
            if not listened_song_features:
                logger.debug(f"User {user_id}: No listened songs found with corresponding features. Skipping profile creation.")
                continue
            user_profile_vector = np.mean(listened_song_features, axis=0)
            user_profiles[user_id] = {'vector': user_profile_vector, 'listened_items': set(user_listened_songs)}
        
        logger.debug(f"Completed building user profiles. Generated {len(user_profiles)} profiles.")
        return user_profiles

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_listened: bool = True,
        include_metadata: bool = True,
        testing_mode: bool = False,
        user_interactions: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError("Recommender not trained. Call .train() first.")
        if testing_mode:
            logger.info("Generating %d recommendations for user %s", n, user_id)

        user_profile = self.user_profiles.get(user_id)
        if user_profile is None and user_interactions is not None and not user_interactions.empty:
            logger.warning(f"User profile for {user_id} not found. Attempting to build on-the-fly.")
            single_user_interactions = user_interactions[user_interactions['user_id'] == user_id]
            if not single_user_interactions.empty:
                temp_user_profiles = self._build_user_profiles(single_user_interactions, self.item_features, self.item_ids)
                if user_id in temp_user_profiles:
                    user_profile = temp_user_profiles[user_id]
                    self.user_profiles[user_id] = user_profile
                    logger.info(f"Successfully built user profile for {user_id} on-the-fly.")
                else:
                    logger.warning(f"Failed to build user profile for {user_id} on-the-fly. Cannot recommend.")
                    return pd.DataFrame()
            else:
                logger.warning(f"No interactions found for user {user_id} in provided user_interactions. Cannot recommend.")
                return pd.DataFrame()
        elif user_profile is None:
            logger.warning(f"User profile for {user_id} not found and no interactions provided. Cannot recommend.")
            return pd.DataFrame()

        user_profile_vector = user_profile['vector']
        user_listened_items = user_profile.get('listened_items', set())

        if self.item_features is None or len(self.item_ids) == 0:
            logger.error("Item features not initialized during training. Cannot generate recommendations.")
            return pd.DataFrame()

        similarities = cosine_similarity(user_profile_vector.reshape(1, -1), self.item_features).flatten()
        if similarities.max() - similarities.min() > 1e-10:
            similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())
        else:
            similarities = np.full_like(similarities, 0.5)

        user_language_weights = calculate_language_weights(
            list(user_listened_items), [1.0] * len(user_listened_items), self.song_to_language
        )
        user_genre_weights = calculate_genre_weights(
            list(user_listened_items), [1.0] * len(user_listened_items), self.song_to_genres
        )

        scores = []
        for idx, item_id in enumerate(self.item_ids):
            if exclude_listened and item_id in user_listened_items:
                scores.append(-1.0)
                continue

            item_weight_factor = calculate_item_weights(
                item_id, self.song_to_tier, self.song_to_pop, self.song_to_fam,
                user_genre_weights, self.song_to_genres,
                self.song_to_language, user_language_weights,
                self.tier_weights
            )

            item_genres = set(self.song_to_genres.get(item_id, []))
            if user_id not in self.user_extended_genres_cache:
                current_user_genres = {genre for song_id in user_listened_items for genre in self.song_to_genres.get(song_id, [])}
                self.user_extended_genres_cache[user_id] = current_user_genres.union(
                    {rel_genre for genre in current_user_genres for rel_genre in self.related_genres_map.get(genre, [genre])}
                )

            user_extended_genres = self.user_extended_genres_cache.get(user_id, set())
            genre_overlap_score = 0.0
            if user_extended_genres and item_genres.intersection(user_extended_genres):
                genre_overlap_score = len(item_genres.intersection(user_extended_genres)) / len(item_genres) if item_genres else 0.0

            item_weight_factor *= (1.0 + genre_overlap_score * 2.0)
            item_weight_factor = np.clip(item_weight_factor, 0.0, 2.0)

            final_score = (
                similarities[idx] * self.feature_weights['audio_features'] +
                item_weight_factor * (self.feature_weights['genres'] + self.feature_weights['language'])
            )
            scores.append(final_score)

        scores = np.array(scores)
        valid_scores_mask = scores >= 0
        if np.any(valid_scores_mask):
            valid_scores = scores[valid_scores_mask]
            if valid_scores.max() - valid_scores.min() > 1e-10:
                scores[valid_scores_mask] = (valid_scores - valid_scores.min()) / (valid_scores.max() - valid_scores.min())
            else:
                scores[valid_scores_mask] = 0.5
        else:
            scores = np.full_like(scores, 0.0)

        ranked = sorted(zip(self.item_ids, scores), key=lambda x: -x[1])
        ranked = [x for x in ranked if x[1] >= 0]
        top_candidate_count = max(n * 5, 20)
        top_n_candidates = ranked[:top_candidate_count]

        recs_df = pd.DataFrame({
            'user_id': user_id,
            'song_id': [x[0] for x in top_n_candidates],
            'score': [x[1] for x in top_n_candidates]
        })

        if include_metadata and not recs_df.empty:
            desired_columns = ['song_id', 'title', 'artist_name', 'top_genre', 'language', 'artist_tier', 'artist_id']
            available_columns = [col for col in desired_columns if col in self.song_metadata.columns]
            if not available_columns or 'song_id' not in available_columns:
                logger.error("Essential metadata columns missing from song_metadata; cannot merge.")
                return recs_df.sort_values('score', ascending=False).head(n)

            if testing_mode:
                logger.info("Merging metadata with columns: %s", available_columns)
            recs_df = recs_df.merge(
                self.song_metadata[available_columns],
                on='song_id',
                how='left'
            )

            for col in ['title', 'artist_name', 'top_genre', 'language', 'artist_tier', 'artist_id']:
                if col in recs_df.columns:
                    if recs_df[col].dtype == 'object':
                        recs_df[col] = recs_df[col].fillna(f'Unknown {col.replace("_", " ").title()}')
                    elif col in ['external_popularity', 'external_familiarity']:
                        recs_df[col] = recs_df[col].fillna(0.0)

            if 'artist_id' in recs_df.columns and 'artist_tier' in recs_df.columns:
                max_exposure = {
                    'emerging_new': 2,
                    'emerging_trending': 2,
                    'rising_established': 2,
                    'mid_tier': 2,
                    'established': 2,
                    'established_trending': 1,
                    'established_legacy': 2
                }
                final_recs = []
                artist_exposure_tracker = {}

                for _, row in recs_df.sort_values('score', ascending=False).iterrows():
                    artist_id = row['artist_id']
                    artist_tier = row['artist_tier']
                    current_exposure = artist_exposure_tracker.get(artist_id, 0)
                    tier_limit = max_exposure.get(artist_tier, 3)

                    if current_exposure < tier_limit:
                        final_recs.append(row)
                        artist_exposure_tracker[artist_id] = current_exposure + 1
                    if len(final_recs) >= n:
                        break

                recs_df = pd.DataFrame(final_recs).head(n)
            else:
                recs_df = recs_df.sort_values('score', ascending=False).head(n)
        else:
            recs_df = recs_df.sort_values('score', ascending=False).head(n)

        if testing_mode and not recs_df.empty:
            logger.info(f"Top {n} recommended song_ids: {recs_df['song_id'].head(n).tolist()}")
        if testing_mode:
            logger.info("Generated %d recommendations for user %s", len(recs_df), user_id)
        return recs_df

    def recommend_similar_items(
        self,
        seed_item_id: str,
        user_id: Optional[str] = None,
        n: int = 10,
        exclude_seed: bool = True,
        include_metadata: bool = True,
        testing_mode: bool = False
    ) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError("Recommender not trained. Call .train() first.")
        if testing_mode:
            logger.info("Generating %d similar items for seed %s with user %s", n, seed_item_id, user_id or "None")

        if not self.item_ids or seed_item_id not in self.item_ids:
            logger.warning(f"Seed item {seed_item_id} not found in trained item_ids; returning empty recommendations.")
            return pd.DataFrame(columns=['seed_item_id', 'song_id', 'score'])

        idx = self.song_id_to_index.get(seed_item_id)
        if idx is None:
            logger.error(f"Seed item ID {seed_item_id} not found in song_id_to_index mapping during recommend_similar_items.")
            return pd.DataFrame()

        seed_genres = set(self.song_to_genres.get(seed_item_id, []))
        seed_top_genre = None
        if self.song_metadata is not None and 'top_genre' in self.song_metadata.columns:
            seed_song_meta = self.song_metadata[self.song_metadata['song_id'] == seed_item_id]
            if not seed_song_meta.empty:
                seed_top_genre = seed_song_meta['top_genre'].iloc[0]

        if testing_mode:
            logger.info(f"Seed song {seed_item_id} genres: {seed_genres}, top_genre: {seed_top_genre}")

        similarities = self.similarity_matrix[idx]
        similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-10)

        scores = similarities.copy()
        user_language_weights = {}
        user_genre_weights = {}
        user_listened_items = set()

        if user_id:
            user_profile_data = self.user_profiles.get(user_id)
            if user_profile_data and 'listened_items' in user_profile_data:
                user_listened_items = user_profile_data['listened_items']
                user_language_weights = calculate_language_weights(
                    list(user_listened_items), [1.0] * len(user_listened_items), self.song_to_language
                )
                user_genre_weights = calculate_genre_weights(
                    list(user_listened_items), [1.0] * len(user_listened_items), self.song_to_genres
                )
            if user_id not in self.user_extended_genres_cache:
                current_user_genres = {genre for song_id in user_listened_items for genre in self.song_to_genres.get(song_id, [])}
                self.user_extended_genres_cache[user_id] = current_user_genres.union(
                    {rel_genre for genre in current_user_genres for rel_genre in self.related_genres_map.get(genre, [genre])}
                )

        extended_genres = set()
        if user_id and user_id in self.user_extended_genres_cache:
            extended_genres = self.user_extended_genres_cache[user_id]
        else:
            for genre in seed_genres:
                extended_genres.update(self.related_genres_map.get(genre, [genre]))
            extended_genres.update(seed_genres)

        if testing_mode:
            logger.info(f"Extended genres for similar items: {len(extended_genres)} entries.")

        non_overlap_count = 0
        for i, item_id in enumerate(self.item_ids):
            if exclude_seed and item_id == seed_item_id:
                scores[i] = -1.0
                continue

            if not all([self.song_to_tier, self.song_to_pop, self.song_to_fam, self.song_to_genres, self.song_to_language, self.tier_weights]):
                logger.warning(f"Required mappings not fully initialized for item weight calculation for item {item_id}. Skipping complex weight calculation.")
                scores[i] = similarities[i] * self.feature_weights['audio_features']
                continue

            item_weight_factor = calculate_item_weights(
                item_id,
                self.song_to_tier,
                self.song_to_pop,
                self.song_to_fam,
                user_genre_weights,
                self.song_to_genres,
                self.song_to_language,
                user_language_weights,
                self.tier_weights
            )

            item_genres = set(self.song_to_genres.get(item_id, []))
            genre_weight_multiplier = 1.0
            if not item_genres.intersection(extended_genres):
                genre_weight_multiplier = 0.0001
                non_overlap_count += 1
                if testing_mode:
                    logger.debug(f"No genre overlap for {item_id}: extended_genres={extended_genres}, item_genres={item_genres}")
            else:
                genre_overlap = len(item_genres.intersection(extended_genres)) / len(item_genres) if item_genres else 0.5
                genre_weight_multiplier = 0.7 + 0.3 * genre_overlap

            if user_id and user_genre_weights:
                user_genres = {genre for song_id in user_listened_items for genre in self.song_to_genres.get(song_id, [])}
                if item_genres.intersection(user_genres):
                    user_overlap_with_item_genres = len(item_genres.intersection(user_genres)) / len(item_genres) if item_genres else 0
                    genre_weight_multiplier *= (1.0 + 1.5 * user_overlap_with_item_genres)

            item_top_genre = None
            if self.song_metadata is not None and 'top_genre' in self.song_metadata.columns:
                item_song_meta = self.song_metadata[self.song_metadata['song_id'] == item_id]
                if not item_song_meta.empty:
                    item_top_genre = item_song_meta['top_genre'].iloc[0]

            if seed_top_genre and item_top_genre == seed_top_genre:
                genre_weight_multiplier *= 1.5

            item_weight_factor *= genre_weight_multiplier
            item_weight_factor = np.clip(item_weight_factor, 0.01, 5.0)

            scores[i] = (
                similarities[i] * self.feature_weights['audio_features'] +
                item_weight_factor * (self.feature_weights['genres'] + self.feature_weights['language'])
            )
            if testing_mode:
                logger.debug(f"Item {item_id}: Score={scores[i]:.4f}, Similarity={similarities[i]:.4f}, ItemWeightFactor={item_weight_factor:.4f}, Genres={item_genres}")

        if testing_mode:
            logger.info(f"Total non-overlap items (with extended genres): {non_overlap_count}")

        valid_scores_mask = scores >= 0
        if np.any(valid_scores_mask):
            valid_scores = scores[valid_scores_mask]
            if valid_scores.max() - valid_scores.min() > 1e-10:
                scores[valid_scores_mask] = (valid_scores - valid_scores.min()) / (valid_scores.max() - valid_scores.min())
            else:
                scores[valid_scores_mask] = 0.5
        else:
            scores = np.full_like(scores, 0.0)

        if testing_mode:
            logger.info(f"Similar item scores range (after final normalization): min={scores[scores>=0].min() if scores[scores>=0].size > 0 else 0.0:.4f}, max={scores[scores>=0].max() if scores[scores>=0].size > 0 else 0.0:.4f}")

        ranked_items = [(self.item_ids[i], scores[i]) for i in range(len(self.item_ids)) if scores[i] >= 0]
        ranked_items.sort(key=lambda x: -x[1])
        top_candidate_count = max(n * 5, 20)
        top_n_candidates = ranked_items[:top_candidate_count]

        recs_df = pd.DataFrame({
            'user_id': user_id,
            'seed_item_id': seed_item_id,
            'song_id': [x[0] for x in top_n_candidates],
            'score': [x[1] for x in top_n_candidates]
        })

        if include_metadata and not recs_df.empty:
            desired_columns = ['song_id', 'title', 'artist_name', 'top_genre', 'language', 'artist_tier', 'artist_id']
            available_columns = [col for col in desired_columns if col in self.song_metadata.columns]
            if not available_columns or 'song_id' not in available_columns:
                logger.error("Essential metadata columns missing from song_metadata; cannot merge metadata for similar items.")
                return recs_df.sort_values('score', ascending=False).head(n)

            if testing_mode:
                logger.info("Merging metadata with columns: %s", available_columns)
            recs_df = recs_df.merge(
                self.song_metadata[available_columns],
                on='song_id',
                how='left'
            )

            for col in ['title', 'artist_name', 'top_genre', 'language', 'artist_tier', 'artist_id']:
                if col in recs_df.columns:
                    if recs_df[col].dtype == 'object':
                        recs_df[col] = recs_df[col].fillna(f'Unknown {col.replace("_", " ").title()}')
                    elif col in ['external_popularity', 'external_familiarity']:
                        recs_df[col] = recs_df[col].fillna(0.0)

            if 'artist_id' in recs_df.columns and 'artist_tier' in recs_df.columns:
                max_exposure = {
                    'emerging_new': 2,
                    'emerging_trending': 2,
                    'rising_established': 2,
                    'mid_tier': 2,
                    'established': 2,
                    'established_trending': 1,
                    'established_legacy': 2
                }
                final_recs = []
                artist_exposure_tracker = {}

                for _, row in recs_df.sort_values('score', ascending=False).iterrows():
                    artist_id = row['artist_id']
                    artist_tier = row['artist_tier']
                    current_exposure = artist_exposure_tracker.get(artist_id, 0)
                    tier_limit = max_exposure.get(artist_tier, 3)

                    if current_exposure < tier_limit:
                        final_recs.append(row)
                        artist_exposure_tracker[artist_id] = current_exposure + 1
                    if len(final_recs) >= n:
                        break

                recs_df = pd.DataFrame(final_recs).head(n)
            else:
                recs_df = recs_df.sort_values('score', ascending=False).head(n)
        else:
            recs_df = recs_df.sort_values('score', ascending=False).head(n)

        if testing_mode:
            logger.debug(f"Generated %d similar items for seed %s", len(recs_df), seed_item_id)
        return recs_df