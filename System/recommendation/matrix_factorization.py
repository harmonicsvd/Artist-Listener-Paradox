import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Optional, List, Set
import logging
import os
import pickle

from System.recommendation.base import RecommenderBase
from System.recommendation.utils.weights import calculate_item_weights, calculate_language_weights, calculate_genre_weights
from System.recommendation.utils.mappings import create_song_to_tier_mapping, create_genre_mapping, create_language_mapping, create_artist_metric_mappings, compute_related_genres
from tqdm import tqdm

logger = logging.getLogger(__name__)

class MatrixFactorizationRecommender(RecommenderBase):
    """Matrix Factorization recommender using SVD for collaborative filtering."""

    def __init__(
        self,
        tier_weights: Dict[str, float],
        name: str = "MatrixFactorization",
        n_factors: int = 50,
        mf_component_weight: float = 1.0,
        item_weight_component_weight: float = 1.5,
        song_to_tier: Optional[Dict[str, str]] = None,
        song_to_pop: Optional[Dict[str, float]] = None,
        song_to_fam: Optional[Dict[str, float]] = None,
        song_to_genres: Optional[Dict[str, List[str]]] = None,
        song_to_language: Optional[Dict[str, str]] = None,
        user_profiles: Optional[Dict[str, Dict]] = None,
        user_extended_genres_cache: Optional[Dict[str, Set[str]]] = None,
        related_genres_map: Optional[Dict[str, Set[str]]] = None,
        song_to_artist: Optional[Dict[str, str]] = None
    ):
        super().__init__(name)
        self.tier_weights = tier_weights
        self.n_factors = n_factors
        self.mf_component_weight = mf_component_weight
        self.item_weight_component_weight = item_weight_component_weight

        self.user_item_matrix_sparse = None
        self.user_latent_factors = None
        self.item_latent_factors = None
        self.user_id_to_index = None
        self.song_id_to_index = None
        self.index_to_song_id = None
        self.index_to_user_id = None
        self.song_metadata = None
        self.artist_identification = None
        self.song_to_artist = song_to_artist

        self.song_to_tier = song_to_tier
        self.song_to_pop = song_to_pop
        self.song_to_fam = song_to_fam
        self.song_to_genres = song_to_genres
        self.song_to_language = song_to_language
        self.user_profiles = user_profiles if user_profiles is not None else {}
        self.user_extended_genres_cache = user_extended_genres_cache if user_extended_genres_cache is not None else {}
        self.related_genres_map = related_genres_map

        self.mf_score_scaler = MinMaxScaler()
        self.item_weight_scaler = MinMaxScaler()

    def _build_all_user_profiles_for_mf(self, user_interactions: pd.DataFrame) -> Dict[str, Dict]:
        """
        Builds user profiles (language/genre weights, listened items) for all users
        during MF training.
        """
        all_user_profiles = {}
        unique_users = user_interactions['user_id'].unique()
        
        logger.info(f"Building initial MF user profiles for {len(unique_users)} users.")
        
        for user_id in tqdm(unique_users, desc="Building MF User Profiles", leave=False, disable=False):
            user_items = user_interactions[user_interactions['user_id'] == user_id]
            if user_items.empty or len(user_items) < 1:
                logger.debug(f"User {user_id} has no interactions or insufficient interactions. Skipping profile creation.")
                continue

            item_ids_listened = user_items['song_id'].tolist()
            play_counts_listened = user_items.get('play_count', pd.Series([1.0] * len(item_ids_listened))).tolist()

            # Ensure essential mappings are available before calculating weights
            if not all([self.song_to_language, self.song_to_genres, self.related_genres_map]):
                logger.warning(f"Essential mappings not fully initialized for user {user_id} profile. Skipping weight calculation for this user.")
                # Create a basic profile without weights if mappings are missing
                all_user_profiles[user_id] = {'listened_items': set(item_ids_listened), 'language_weights': {}, 'genre_weights': {}}
                continue

            language_weights = calculate_language_weights(item_ids_listened, play_counts_listened, self.song_to_language)
            genre_weights = calculate_genre_weights(item_ids_listened, play_counts_listened, self.song_to_genres)
            
            user_genres = {genre for item_id in item_ids_listened for genre in self.song_to_genres.get(item_id, [])}
            related_genres = {rel_genre for genre in user_genres for rel_genre in self.related_genres_map.get(genre, {genre})} # Use set literal for single genre
            extended_genres = user_genres.union(related_genres)
            
            all_user_profiles[user_id] = {
                'language_weights': language_weights,
                'genre_weights': genre_weights,
                'listened_items': set(item_ids_listened)
            }
            self.user_extended_genres_cache[user_id] = extended_genres # Cache for recommend calls

        logger.info(f"Completed building initial MF user profiles. Generated {len(all_user_profiles)} profiles.")
        return all_user_profiles

    def train(
        self,
        user_interactions: pd.DataFrame,
        song_features: Optional[pd.DataFrame] = None,
        song_metadata: Optional[pd.DataFrame] = None,
        artist_identification: Optional[pd.DataFrame] = None,
        testing_mode: bool = False,
        cache_dir: Optional[str] = None
    ) -> None:
        logger.info("Training %s recommender", self.name)
        super().train(user_interactions)

        self.song_metadata = song_metadata
        self.artist_identification = artist_identification

        if song_metadata is not None:
            if self.song_to_artist is None:
                self.song_to_artist = dict(zip(song_metadata['song_id'], song_metadata.get('artist_name', pd.Series(index=song_metadata['song_id'], dtype=str))))
            if self.related_genres_map is None and self.song_to_genres: 
                logger.info("Computing related genres map for MatrixFactorization (during train, if not pre-passed)")
                self.related_genres_map = compute_related_genres(self.song_to_genres)
                logger.info("Computed related genres for %d genres", len(self.related_genres_map))
            elif self.related_genres_map is None:
                logger.warning("Cannot compute related_genres_map: song_to_genres is not set.")
        
        if not all([self.song_to_tier, self.song_to_pop, self.song_to_fam, 
                    self.song_to_genres, self.song_to_language, self.related_genres_map, self.tier_weights]):
            logger.error("MatrixFactorization requires essential mappings; missing mappings may cause recommendation errors.")

        actual_num_users = user_interactions['user_id'].nunique()
        actual_num_songs = user_interactions['song_id'].nunique()

        if cache_dir:
            cache_user_factors_path = os.path.join(cache_dir, f"{self.name.lower()}_user_factors_u{actual_num_users}_s{actual_num_songs}_f{self.n_factors}.npy")
            cache_item_factors_path = os.path.join(cache_dir, f"{self.name.lower()}_item_factors_u{actual_num_users}_s{actual_num_songs}_f{self.n_factors}.npy")
            cache_user_idx_map_path = os.path.join(cache_dir, f"{self.name.lower()}_user_id_to_index_u{actual_num_users}_s{actual_num_songs}.pkl")
            cache_song_idx_map_path = os.path.join(cache_dir, f"{self.name.lower()}_song_id_to_index_u{actual_num_users}_s{actual_num_songs}.pkl")
            cache_idx_song_map_path = os.path.join(cache_dir, f"{self.name.lower()}_index_to_song_id_u{actual_num_users}_s{actual_num_songs}.pkl")
            cache_user_profiles_path = os.path.join(cache_dir, f"{self.name.lower()}_user_profiles_users{actual_num_users}.pkl") # New cache path for profiles

            if os.path.exists(cache_user_factors_path) and \
               os.path.exists(cache_item_factors_path) and \
               os.path.exists(cache_user_idx_map_path) and \
               os.path.exists(cache_song_idx_map_path) and \
               os.path.exists(cache_idx_song_map_path) and \
               os.path.exists(cache_user_profiles_path): # Check for user profiles cache
                
                logger.info(f"Attempting to load {self.name} matrices and mappings from cache (users:{actual_num_users}, songs:{actual_num_songs}, factors:{self.n_factors}).")
                try:
                    self.user_latent_factors = np.load(cache_user_factors_path)
                    self.item_latent_factors = np.load(cache_item_factors_path)
                    with open(cache_user_idx_map_path, 'rb') as f:
                        self.user_id_to_index = pickle.load(f)
                    with open(cache_song_idx_map_path, 'rb') as f:
                        self.song_id_to_index = pickle.load(f)
                    with open(cache_idx_song_map_path, 'rb') as f:
                        self.index_to_song_id = pickle.load(f)
                    with open(cache_user_profiles_path, 'rb') as f: # Load user profiles
                        self.user_profiles = pickle.load(f)
                    self.index_to_user_id = {v: k for k, v in self.user_id_to_index.items()}
                    logger.info(f"{self.name} matrices and mappings loaded successfully from cache.")
                    
                    # Verify user_profiles after loading from cache
                    if not self.user_profiles:
                        logger.warning("Loaded user_profiles from cache but it is empty. This may indicate a prior issue.")
                    else:
                        logger.info(f"Loaded {len(self.user_profiles)} user profiles from cache.")

                    self.is_trained = True
                    return
                except Exception as e:
                    logger.warning(f"Error loading {self.name} matrices/profiles from cache: {e}. Falling back to recalculation...")

        logger.info(f"Cache not found or failed to load for {self.name} matrices. Calculating (users:{actual_num_users}, songs:{actual_num_songs}, factors:{self.n_factors})...")

        unique_users = user_interactions['user_id'].unique()
        unique_songs = user_interactions['song_id'].unique()
        
        self.user_id_to_index = {uid: idx for idx, uid in enumerate(unique_users)}
        self.song_id_to_index = {sid: idx for idx, sid in enumerate(unique_songs)}
        self.index_to_song_id = {idx: sid for sid, idx in self.song_id_to_index.items()}
        self.index_to_user_id = {idx: uid for idx, uid in enumerate(unique_users)}

        rows = [self.user_id_to_index[uid] for uid in user_interactions['user_id']]
        cols = [self.song_id_to_index[sid] for sid in user_interactions['song_id']]
        data = user_interactions.get('play_count', pd.Series(np.ones(len(user_interactions)))).values.astype(np.float32)
        
        self.user_item_matrix_sparse = csr_matrix((data, (rows, cols)), shape=(len(unique_users), len(unique_songs)))
        
        k_factors = min(self.n_factors, min(self.user_item_matrix_sparse.shape) - 1)
        if k_factors <= 0:
            logger.error("Not enough data to perform SVD with given n_factors. Skipping training.")
            return

        logger.info("Performing SVD with %d factors...", k_factors)
        U, s, Vt = svds(self.user_item_matrix_sparse, k=k_factors)
        
        s_diag_matrix = np.diag(s)
        self.user_latent_factors = U @ s_diag_matrix
        self.item_latent_factors = Vt.T
        
        logger.info("User latent factors shape: %s, Item latent factors shape: %s", 
                    self.user_latent_factors.shape, self.item_latent_factors.shape)

        # Build user profiles (genre/language weights, listened items) here during train
        self.user_profiles = self._build_all_user_profiles_for_mf(user_interactions)
        logger.debug(f"After MatrixFactorization train, user_profiles has {len(self.user_profiles)} entries.")

        self.is_trained = True
        
        if testing_mode:
            logger.info(f"{self.name} training complete.")

        if cache_dir and self.is_trained:
            logger.info(f"Saving {self.name} matrices and mappings to cache for U:{actual_num_users}, S:{actual_num_songs}, F:{self.n_factors}.")
            try:
                os.makedirs(cache_dir, exist_ok=True) # Ensure cache directory exists
                np.save(cache_user_factors_path, self.user_latent_factors)
                np.save(cache_item_factors_path, self.item_latent_factors)
                with open(cache_user_idx_map_path, 'wb') as f:
                    pickle.dump(self.user_id_to_index, f)
                with open(cache_song_idx_map_path, 'wb') as f:
                    pickle.dump(self.song_id_to_index, f)
                with open(cache_idx_song_map_path, 'wb') as f:
                    pickle.dump(self.index_to_song_id, f)
                with open(cache_user_profiles_path, 'wb') as f: # Save user profiles
                    pickle.dump(self.user_profiles, f)
                logger.info(f"{self.name} matrices and mappings saved to cache.")
            except Exception as e:
                logger.error(f"Error saving cache: {e}")

    def _get_user_profile_for_recommendation(self, user_id: str, user_interactions: Optional[pd.DataFrame], testing_mode: bool):
        # First, try to retrieve from the pre-built user_profiles cache
        user_profile = self.user_profiles.get(user_id)
        
        # If not found in cache and user_interactions are provided (for new users/on-the-fly)
        if not user_profile and user_interactions is not None:
            if testing_mode:
                logger.info("User profile for %s not found in pre-built cache. Attempting to build on-the-fly.", user_id)
            user_items = user_interactions[user_interactions['user_id'] == user_id]
            if not user_items.empty and len(user_items) >= 1:
                item_ids = user_items['song_id'].tolist()
                play_counts = user_items.get('play_count', pd.Series([1.0] * len(item_ids))).tolist()
                
                if not all([self.song_to_language, self.song_to_genres, self.related_genres_map]):
                    logger.error("Required mappings not initialized for user profile creation in MF recommend (on-the-fly). Cannot build profile.")
                    return None

                language_weights = calculate_language_weights(item_ids, play_counts, self.song_to_language)
                genre_weights = calculate_genre_weights(item_ids, play_counts, self.song_to_genres)
                
                user_genres = {genre for item_id in item_ids for genre in self.song_to_genres.get(item_id, [])}
                related_genres = {rel_genre for genre in user_genres for rel_genre in self.related_genres_map.get(genre, {genre})}
                extended_genres = user_genres.union(related_genres)

                user_profile = {
                    'language_weights': language_weights,
                    'genre_weights': genre_weights,
                    'listened_items': set(item_ids)
                }
                self.user_profiles[user_id] = user_profile # Cache this newly built profile
                self.user_extended_genres_cache[user_id] = extended_genres
                logger.debug(f"Successfully built on-the-fly profile for user {user_id}.")
            else:
                logger.warning(f"No sufficient interactions found for user {user_id} to build on-the-fly profile; will use default weights if profile is required.")
        elif not user_profile: # If user_profile is still None and no interactions for on-the-fly
            logger.warning(f"User profile for {user_id} not found in cache and no interactions provided. Returning None profile.")
            return None # Return None if no profile can be found or built

        return user_profile

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_listened: bool = True, 
        include_metadata: bool = True,
        testing_mode: bool = False,
        user_interactions: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        if not self.is_trained or self.user_latent_factors is None or self.item_latent_factors is None:
            logger.error(f"Recommender '{self.name}' is not trained or has no valid latent matrices.")
            return pd.DataFrame(columns=['user_id', 'song_id', 'score'])
        
        if testing_mode:
            logger.debug("Generating %d recommendations for user %s", n, user_id)
        
        user_idx = self.user_id_to_index.get(user_id)
        if user_idx is None:
            logger.warning(f"User {user_id} not found in trained matrix; returning empty recommendations.")
            return pd.DataFrame(columns=['user_id', 'song_id', 'score'])
        
        user_vector = self.user_latent_factors[user_idx, :]
        raw_mf_scores = user_vector @ self.item_latent_factors.T
        
        if raw_mf_scores.ndim > 1:
            raw_mf_scores = raw_mf_scores.flatten()

        if raw_mf_scores.size > 0:
            self.mf_score_scaler.fit(raw_mf_scores.reshape(-1, 1))
            normalized_mf_scores = self.mf_score_scaler.transform(raw_mf_scores.reshape(-1, 1)).flatten()
        else:
            normalized_mf_scores = np.zeros_like(raw_mf_scores)

        user_profile = self._get_user_profile_for_recommendation(user_id, user_interactions, testing_mode)
        if not user_profile: # if profile is None after lookup/on-the-fly attempt
            logger.warning(f"No user profile found or built for user {user_id} for item weighting; using only MF scores.")
            # Provide a minimal default profile to avoid further errors in calculations
            user_profile = {'genre_weights': {}, 'language_weights': {}, 'listened_items': set()} 

        # Initialize final scores array with MF scores
        hybrid_scores_final_values = np.copy(normalized_mf_scores)
        item_weights_list = [0.0] * len(self.index_to_song_id) # Initialize with zeros for all songs

        listened_songs_indices = {self.song_id_to_index[sid] for sid in user_profile['listened_items'] if sid in self.song_id_to_index}

        for idx in range(len(self.index_to_song_id)):
            song_id = self.index_to_song_id[idx]
            
            # Apply exclusion directly by setting score to -1.0
            if exclude_listened and idx in listened_songs_indices:
                hybrid_scores_final_values[idx] = -1.0 # Mark as excluded
                item_weights_list[idx] = 0.0 # Set dummy weight for excluded item
                continue
            
            # Ensure required mappings for item weight calculation are present.
            if not all([self.song_to_tier, self.song_to_pop, self.song_to_fam, self.song_to_genres, self.song_to_language, self.tier_weights]):
                logger.warning(f"Required mappings not initialized for item weight calculation for song {song_id} in MF recommend. Skipping item weighting.")
                item_weight = 0.5 # Default neutral weight
            else:
                item_weight = calculate_item_weights(
                    song_id,
                    self.song_to_tier,
                    self.song_to_pop,
                    self.song_to_fam,
                    user_profile['genre_weights'],
                    self.song_to_genres,
                    self.song_to_language,
                    user_profile['language_weights'],
                    self.tier_weights
                )
            item_weights_list[idx] = item_weight # Assign calculated item weight

        # Handle scaling for item_weights_list
        # Only fit scaler on non-negative weights from items that are not excluded
        valid_item_weights_for_scaling = [w for i, w in enumerate(item_weights_list) if hybrid_scores_final_values[i] >= 0]
        
        if valid_item_weights_for_scaling and (max(valid_item_weights_for_scaling) - min(valid_item_weights_for_scaling)) > 1e-10:
            self.item_weight_scaler.fit(np.array(valid_item_weights_for_scaling).reshape(-1, 1))
            # Transform all item_weights_list, including those marked for exclusion (they'll be handled by the -1.0 check later)
            normalized_item_weights = self.item_weight_scaler.transform(np.array(item_weights_list).reshape(-1, 1)).flatten()
        else: # All weights are uniform or list is empty
            normalized_item_weights = np.zeros_like(item_weights_list)

        # Combine the scores only for non-excluded items in a second loop
        for idx in range(len(self.index_to_song_id)):
            if hybrid_scores_final_values[idx] == -1.0: # If item was marked for exclusion, skip it
                continue
            
            final_score = (
                normalized_mf_scores[idx] * self.mf_component_weight +
                normalized_item_weights[idx] * self.item_weight_component_weight
            )
            hybrid_scores_final_values[idx] = final_score

        hybrid_scores = hybrid_scores_final_values # Assign the final computed scores
        
        valid_scores_mask = hybrid_scores >= 0
        valid_scores = hybrid_scores[valid_scores_mask]

        if valid_scores.size == 0 or (valid_scores.max() - valid_scores.min()) < 1e-10:
            logger.warning(f"Final scores for user {user_id} are uniform or empty after exclusion. Cannot normalize meaningfully. Returning empty recommendations.")
            return pd.DataFrame(columns=['user_id', 'song_id', 'score'])
        
        hybrid_scores[valid_scores_mask] = (valid_scores - valid_scores.min()) / (valid_scores.max() - valid_scores.min() + 1e-10)
        
        ranked_song_indices = np.argsort(-hybrid_scores)
        
        top_n_songs_info = []
        top_candidate_count = max(n * 5, 20)

        for i in ranked_song_indices:
            song_id = self.index_to_song_id[i]
            if hybrid_scores[i] < 0:
                continue
            
            top_n_songs_info.append({
                'user_id': user_id,
                'song_id': song_id,
                'score': hybrid_scores[i]
            })
            if len(top_n_songs_info) >= top_candidate_count:
                break

        recs_df = pd.DataFrame(top_n_songs_info)

        if include_metadata and not recs_df.empty:
            desired_columns = ['song_id', 'title', 'artist_name', 'top_genre', 'language', 'artist_id', 'artist_tier']
            available_columns = [col for col in desired_columns if col in self.song_metadata.columns]
            if 'song_id' not in available_columns:
                logger.error("song_id missing from song_metadata; cannot merge metadata for recommendations.")
                return recs_df.sort_values('score', ascending=False).head(n)

            if testing_mode:
                logger.info("Merging metadata with columns: %s", available_columns)
            recs_df = recs_df.merge(
                self.song_metadata[available_columns],
                on='song_id',
                how='left'
            )
            
            # Fill NaNs for safety after merge
            for col in ['title', 'artist_name', 'top_genre', 'language', 'artist_tier', 'artist_id']:
                if col in recs_df.columns:
                    if recs_df[col].dtype == 'object':
                        recs_df[col] = recs_df[col].fillna(f'Unknown {col.replace("_", " ").title()}')
                    elif col in ['external_popularity', 'external_familiarity']: # Example for numeric columns
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
            logger.debug("Generated %d recommendations for user %s", len(recs_df), user_id)
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
        if not self.is_trained or self.item_latent_factors is None:
            logger.error(f"Recommender '{self.name}' is not trained or has no valid item latent factors.")
            return pd.DataFrame(columns=['seed_item_id', 'song_id', 'score'])
        
        if testing_mode:
            logger.debug("Generating %d similar items for seed %s", n, seed_item_id)
        
        seed_idx = self.song_id_to_index.get(seed_item_id)
        if seed_idx is None:
            logger.warning(f"Seed item {seed_item_id} not found in trained matrix; returning empty recommendations.")
            return pd.DataFrame(columns=['seed_item_id', 'song_id', 'score'])
        
        seed_item_vector = self.item_latent_factors[seed_idx, :]
        
        item_similarities = (self.item_latent_factors @ seed_item_vector.T) / (
            np.linalg.norm(self.item_latent_factors, axis=1) * np.linalg.norm(seed_item_vector) + 1e-10
        )
        item_similarities = item_similarities.flatten()

        if item_similarities.size > 0:
            self.mf_score_scaler.fit(item_similarities.reshape(-1, 1))
            normalized_mf_similarities = self.mf_score_scaler.transform(item_similarities.reshape(-1, 1)).flatten()
        else:
            normalized_mf_similarities = np.zeros_like(item_similarities)

        user_profile = None
        extended_genres = set()
        
        # In recommend_similar_items, user_interactions will not be available
        # so we rely only on the pre-built user_profiles cache.
        if user_id:
            user_profile = self.user_profiles.get(user_id)
            if user_profile and user_id in self.user_extended_genres_cache:
                extended_genres = self.user_extended_genres_cache[user_id]
        
        # If no user profile found (either no user_id or not in cache), create a default one
        if not user_profile:
            user_profile = {'genre_weights': {}, 'language_weights': {}, 'listened_items': set()}
            # If no user, base extended_genres on the seed item for genre boost
            if not extended_genres: # Only if extended_genres wasn't set by a user profile
                seed_genres = set(self.song_to_genres.get(seed_item_id, []))
                for genre in seed_genres:
                    extended_genres.update(self.related_genres_map.get(genre, {genre}))
                extended_genres.update(seed_genres)
                if testing_mode:
                    logger.debug(f"No user profile, basing extended genres on seed {seed_item_id}; extended genres size: {len(extended_genres)}")


        hybrid_scores_initial = np.copy(normalized_mf_similarities) # Initialize
        item_weights_list = [0.0] * len(self.index_to_song_id) # Initialize for all songs
        
        for idx in range(len(self.index_to_song_id)):
            song_id = self.index_to_song_id[idx]
            
            if exclude_seed and song_id == seed_item_id:
                hybrid_scores_initial[idx] = -1.0 # Mark as excluded
                item_weights_list[idx] = 0.0 # Dummy weight for excluded item
                continue
            
            # Ensure required mappings for item weight calculation are present.
            if not all([self.song_to_tier, self.song_to_pop, self.song_to_fam, self.song_to_genres, self.song_to_language, self.tier_weights]):
                logger.warning(f"Required mappings not initialized for item weight calculation for song {song_id} in MF similar items. Skipping item weighting.")
                item_weight = 0.5 # Default neutral weight
            else:
                # Apply seed genre boost by checking if song's genres are in extended_genres
                song_genres = set(self.song_to_genres.get(song_id, []))
                genre_match_boost = 0.0
                if extended_genres and song_genres.intersection(extended_genres):
                    genre_match_boost = 1.0 # Simple boost if any overlap
                
                item_weight = calculate_item_weights(
                    song_id,
                    self.song_to_tier,
                    self.song_to_pop,
                    self.song_to_fam,
                    user_profile['genre_weights'],
                    self.song_to_genres,
                    self.song_to_language,
                    user_profile['language_weights'],
                    self.tier_weights
                ) 
                
                # Apply additional boost based on genre match
                item_weight *= (1 + genre_match_boost * 0.5) # Example: 50% boost if genre matches

            item_weights_list[idx] = item_weight # Assign calculated item weight

        valid_item_weights = [w for i, w in enumerate(item_weights_list) if hybrid_scores_initial[i] >= 0]
        if valid_item_weights and (max(valid_item_weights) - min(valid_item_weights)) > 1e-10:
            self.item_weight_scaler.fit(np.array(valid_item_weights).reshape(-1, 1))
            normalized_item_weights = self.item_weight_scaler.transform(np.array(item_weights_list).reshape(-1, 1)).flatten()
        else:
            normalized_item_weights = np.zeros_like(item_weights_list)

        hybrid_scores_final_values = np.copy(hybrid_scores_initial) # Start with current state (incl. exclusions)

        for idx in range(len(self.index_to_song_id)):
            if hybrid_scores_final_values[idx] == -1.0: # If excluded, skip
                continue
            
            final_score = (
                normalized_mf_similarities[idx] * self.mf_component_weight +
                normalized_item_weights[idx] * self.item_weight_component_weight
            )
            hybrid_scores_final_values[idx] = final_score
            
        hybrid_scores = hybrid_scores_final_values
        
        valid_scores_mask = hybrid_scores >= 0
        valid_scores = hybrid_scores[valid_scores_mask]

        if valid_scores.size == 0 or (valid_scores.max() - valid_scores.min()) < 1e-10:
            logger.warning(f"Final scores for seed {seed_item_id} are uniform or empty after exclusion. Cannot normalize meaningfully. Returning empty recommendations.")
            return pd.DataFrame(columns=['seed_item_id', 'song_id', 'score']) 
        
        hybrid_scores[valid_scores_mask] = (valid_scores - valid_scores.min()) / (valid_scores.max() - valid_scores.min() + 1e-10)
        
        ranked_song_indices = np.argsort(-hybrid_scores)
        
        top_n_songs_info = []
        top_candidate_count = max(n * 5, 20)

        for i in ranked_song_indices:
            song_id = self.index_to_song_id[i]
            if hybrid_scores[i] < 0:
                continue
            
            top_n_songs_info.append({
                'seed_item_id': seed_item_id,
                'song_id': song_id,
                'score': hybrid_scores[i]
            })
            if len(top_n_songs_info) >= top_candidate_count:
                break
        
        recs_df = pd.DataFrame(top_n_songs_info)
        
        if include_metadata and not recs_df.empty:
            desired_columns = ['song_id', 'title', 'artist_name', 'top_genre', 'language', 'artist_id', 'artist_tier']
            available_columns = [col for col in desired_columns if col in self.song_metadata.columns]
            if 'song_id' not in available_columns:
                logger.error("song_id missing from song_metadata; cannot merge metadata for similar items.")
                return recs_df.sort_values('score', ascending=False).head(n)
            
            if testing_mode:
                logger.info("Merging metadata with columns: %s", available_columns)
            recs_df = recs_df.merge(
                self.song_metadata[available_columns],
                on='song_id',
                how='left'
            )
            
            # Fill NaNs for safety after merge
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
        
        if testing_mode:
            logger.debug(f"Generated %d similar items for seed %s", len(recs_df), seed_item_id)
        return recs_df
