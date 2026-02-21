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
from System.recommendation.utils.weights import calculate_item_weights, calculate_genre_weights # Removed calculate_language_weights
from System.recommendation.utils.mappings import create_song_to_tier_mapping, create_genre_mapping, create_language_mapping, create_artist_metric_mappings, compute_related_genres
from tqdm import tqdm

logger = logging.getLogger(__name__)

class MatrixFactorizationRecommender(RecommenderBase):
    """Matrix Factorization recommender using SVD for collaborative filtering."""

    def __init__(
        self,
        tier_weights: Dict[str, float],
        name: str = "MatrixFactorization",
        n_factors: int = 100,
        mf_component_weight: float = 3.8,
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
        Builds user profiles (genre weights, listened items) for all users
        during MF training. Language weights are no longer calculated here.
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
            # Removed self.song_to_language from this check as language is handled by filtering
            if not all([self.song_to_genres, self.related_genres_map]): 
                logger.warning(f"Essential mappings not fully initialized for user {user_id} profile. Skipping genre weight calculation for this user.")
                # Create a basic profile without weights if mappings are missing
                all_user_profiles[user_id] = {'listened_items': set(item_ids_listened), 'genre_weights': {}}
                continue

            # language_weights removed from calculation and storage
            genre_weights = calculate_genre_weights(item_ids_listened, play_counts_listened, self.song_to_genres)
            
            user_genres = {genre for item_id in item_ids_listened for genre in self.song_to_genres.get(item_id, [])}
            related_genres = {rel_genre for genre in user_genres for rel_genre in self.related_genres_map.get(genre, {genre})} # Use set literal for single genre
            extended_genres = user_genres.union(related_genres)
            
            all_user_profiles[user_id] = {
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
            
            # Ensure song_to_language is populated during train
            if self.song_to_language is None and 'language' in song_metadata.columns:
                self.song_to_language = dict(zip(song_metadata['song_id'], song_metadata['language']))

        # Removed self.song_to_language from this check as language is handled by filtering
        if not all([self.song_to_tier, self.song_to_pop, self.song_to_fam, 
                    self.song_to_genres, self.related_genres_map, self.tier_weights]):
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
                
                # Removed self.song_to_language from this check as language is handled by filtering
                if not all([self.song_to_genres, self.related_genres_map]): 
                    logger.error("Required mappings not initialized for user profile creation in MF recommend (on-the-fly). Cannot build profile.")
                    return None

                # language_weights removed from calculation and storage
                genre_weights = calculate_genre_weights(item_ids, play_counts, self.song_to_genres)
                
                user_genres = {genre for item_id in item_ids for genre in self.song_to_genres.get(item_id, [])}
                related_genres = {rel_genre for genre in user_genres for rel_genre in self.related_genres_map.get(genre, {genre})}
                extended_genres = user_genres.union(related_genres)

                user_profile = {
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

        user_profile = self._get_user_profile_for_recommendation(user_id, user_interactions, testing_mode)
        if not user_profile: # if profile is None after lookup/on-the-fly attempt
            logger.warning(f"No user profile found or built for user {user_id} for item weighting; using only MF scores.")
            # Provide a minimal default profile to avoid further errors in calculations
            user_profile = {'genre_weights': {}, 'listened_items': set()} # Removed language_weights

        # Determine dominant language(s) from user's listening history
        user_listened_items = user_profile['listened_items']
        dominant_languages = []
        if user_listened_items and self.song_to_language:
            listened_languages = [self.song_to_language.get(song_id) for song_id in user_listened_items if self.song_to_language.get(song_id)]
            if listened_languages:
                language_counts = pd.Series(listened_languages).value_counts()
                max_count = language_counts.max()
                dominant_languages = language_counts[language_counts == max_count].index.tolist()
                if testing_mode:
                    logger.info(f"User {user_id} dominant language(s): {dominant_languages}")
            else:
                logger.debug(f"No language information found for listened songs of user {user_id}.")
        else:
            logger.debug(f"No listened items or song_to_language mapping for user {user_id}.")

        # Prepare a list of candidate song_ids and their original indices
        candidate_song_ids = []
        candidate_original_indices = []

        if dominant_languages:
            for i in range(len(self.index_to_song_id)):
                song_id = self.index_to_song_id[i]
                if self.song_to_language and self.song_to_language.get(song_id) in dominant_languages:
                    candidate_song_ids.append(song_id)
                    candidate_original_indices.append(i)
            
            if not candidate_song_ids: # Fallback if no songs match dominant language
                logger.warning(f"No songs found matching dominant language(s) {dominant_languages}. Reverting to all songs for recommendation calculation.")
                candidate_song_ids = [self.index_to_song_id[i] for i in range(len(self.index_to_song_id))]
                candidate_original_indices = list(range(len(self.index_to_song_id)))
        else:
            # If no dominant language found, consider all songs
            candidate_song_ids = [self.index_to_song_id[i] for i in range(len(self.index_to_song_id))]
            candidate_original_indices = list(range(len(self.index_to_song_id)))

        # Extract MF scores only for the candidate songs
        mf_scores_for_candidates = raw_mf_scores[candidate_original_indices]

        # Handle cases where mf_scores_for_candidates might be uniform or empty
        if mf_scores_for_candidates.size == 0 or (mf_scores_for_candidates.max() - mf_scores_for_candidates.min()) < 1e-10:
            logger.warning(f"MF scores for user {user_id} (filtered by language) are uniform or empty. Setting normalized scores to 0.5.")
            normalized_mf_scores = np.full_like(mf_scores_for_candidates, 0.5) # Assign a neutral score
            # If no candidates, return empty DataFrame
            if mf_scores_for_candidates.size == 0:
                return pd.DataFrame(columns=['user_id', 'song_id', 'score'])
        else:
            # Normalize MF scores for candidates
            self.mf_score_scaler.fit(mf_scores_for_candidates.reshape(-1, 1))
            normalized_mf_scores = self.mf_score_scaler.transform(mf_scores_for_candidates.reshape(-1, 1)).flatten()

        hybrid_scores_final_values = np.copy(normalized_mf_scores)
        item_weights_list = [0.0] * len(candidate_song_ids) # Initialize with zeros for candidate songs

        listened_songs_indices = {self.song_id_to_index[sid] for sid in user_profile['listened_items'] if sid in self.song_id_to_index}

        for idx_in_candidate, song_id in enumerate(candidate_song_ids):
            original_idx = self.song_id_to_index[song_id] # Get original index for exclusion check
            
            # Apply exclusion directly by setting score to -1.0
            if exclude_listened and original_idx in listened_songs_indices:
                hybrid_scores_final_values[idx_in_candidate] = -1.0 # Mark as excluded
                item_weights_list[idx_in_candidate] = 0.0 # Set dummy weight for excluded item
                continue
            
            # Ensure required mappings for item weight calculation are present.
            # Removed self.song_to_language from this check as language is handled by filtering
            if not all([self.song_to_tier, self.song_to_pop, self.song_to_fam, self.song_to_genres, self.tier_weights]): 
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
                    None, None, # Language weights are no longer passed here
                    self.tier_weights
                )
            item_weights_list[idx_in_candidate] = item_weight # Assign calculated item weight

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
        for idx_in_candidate in range(len(candidate_song_ids)):
            if hybrid_scores_final_values[idx_in_candidate] == -1.0: # If item was marked for exclusion, skip it
                continue
            
            final_score = (
                normalized_mf_scores[idx_in_candidate] * self.mf_component_weight +
                normalized_item_weights[idx_in_candidate] * self.item_weight_component_weight
            )
            hybrid_scores_final_values[idx_in_candidate] = final_score

        hybrid_scores = hybrid_scores_final_values # Assign the final computed scores
        
        valid_scores_mask = hybrid_scores >= 0
        valid_scores = hybrid_scores[valid_scores_mask]

        if valid_scores.size == 0 or (valid_scores.max() - valid_scores.min()) < 1e-10:
            logger.warning(f"Final scores for user {user_id} are uniform or empty after exclusion. Cannot normalize meaningfully. Returning empty recommendations.")
            return pd.DataFrame(columns=['user_id', 'song_id', 'score'])
        
        # Ensure normalization only happens if there's a range to normalize over
        if valid_scores.max() - valid_scores.min() > 1e-10:
            hybrid_scores[valid_scores_mask] = (valid_scores - valid_scores.min()) / (valid_scores.max() - valid_scores.min() + 1e-10)
        else:
            # If all valid scores are identical, set them to a neutral value (e.g., 0.5)
            hybrid_scores[valid_scores_mask] = 0.5
        
        ranked_song_indices_in_scores = np.argsort(-hybrid_scores) # These are indices within the `hybrid_scores` array (which corresponds to `candidate_song_ids`)
        
        top_n_songs_info = []
        top_candidate_count = max(n * 5, 20)

        for i_in_scores in ranked_song_indices_in_scores:
            song_id = candidate_song_ids[i_in_scores] # Get actual song_id from the candidate list
            if hybrid_scores[i_in_scores] < 0:
                continue
            
            top_n_songs_info.append({
                'user_id': user_id,
                'song_id': song_id,
                'score': hybrid_scores[i_in_scores]
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
            user_profile = {'genre_weights': {}, 'listened_items': set()} # Removed language_weights
            # If no user, base extended_genres on the seed item for genre boost
            if not extended_genres: # Only if extended_genres wasn't set by a user profile
                seed_genres = set(self.song_to_genres.get(seed_item_id, []))
                for genre in seed_genres:
                    extended_genres.update(self.related_genres_map.get(genre, {genre}))
                extended_genres.update(seed_genres)
                if testing_mode:
                    logger.debug(f"No user profile, basing extended genres on seed {seed_item_id}; extended genres size: {len(extended_genres)}")

        # Determine target language(s) for similar items
        target_languages = []
        user_listened_items = set()

        if user_id and self.song_to_language:
            # Get user's listened items to determine dominant language
            if user_profile and 'listened_items' in user_profile:
                user_listened_items = user_profile['listened_items']
            
            if user_listened_items:
                listened_languages = [self.song_to_language.get(song_id) for song_id in user_listened_items if self.song_to_language.get(song_id)]
                if listened_languages:
                    language_counts = pd.Series(listened_languages).value_counts()
                    max_count = language_counts.max()
                    target_languages = language_counts[language_counts == max_count].index.tolist()
                    if testing_mode:
                        logger.info(f"User {user_id} dominant language(s) for similar items: {target_languages}")
        
        if not target_languages and self.song_to_language: # If no user or no dominant user language, use seed item's language
            seed_language = self.song_to_language.get(seed_item_id)
            if seed_language:
                target_languages = [seed_language]
                if testing_mode:
                    logger.info(f"Using seed item's language {seed_language} for similar items.")
            else:
                logger.warning(f"No language found for seed item {seed_item_id}. Similar items will not be language filtered.")
        
        # Prepare a list of candidate song_ids and their original indices
        candidate_song_ids = []
        candidate_original_indices = []

        if target_languages:
            for i in range(len(self.index_to_song_id)):
                song_id = self.index_to_song_id[i]
                if self.song_to_language and self.song_to_language.get(song_id) in target_languages:
                    candidate_song_ids.append(song_id)
                    candidate_original_indices.append(i)
            if not candidate_song_ids:
                logger.warning(f"No similar items found matching target language(s) {target_languages}. Reverting to all songs for similarity calculation.")
                candidate_song_ids = [self.index_to_song_id[i] for i in range(len(self.index_to_song_id))]
                candidate_original_indices = list(range(len(self.index_to_song_id)))
        else:
            candidate_song_ids = [self.index_to_song_id[i] for i in range(len(self.index_to_song_id))]
            candidate_original_indices = list(range(len(self.index_to_song_id)))

        # Extract similarities only for the candidate songs
        item_similarities_for_candidates = item_similarities[candidate_original_indices] # Use the already calculated item_similarities (all)

        # Handle cases where item_similarities_for_candidates might be uniform or empty
        if item_similarities_for_candidates.size == 0 or (item_similarities_for_candidates.max() - item_similarities_for_candidates.min()) < 1e-10:
            logger.warning(f"MF similarities for seed {seed_item_id} (filtered by language) are uniform or empty. Setting normalized similarities to 0.5.")
            normalized_mf_similarities = np.full_like(item_similarities_for_candidates, 0.5) # Assign a neutral similarity
            # If no candidates, return empty DataFrame
            if item_similarities_for_candidates.size == 0:
                return pd.DataFrame(columns=['seed_item_id', 'song_id', 'score'])
        else:
            self.mf_score_scaler.fit(item_similarities_for_candidates.reshape(-1, 1))
            normalized_mf_similarities = self.mf_score_scaler.transform(item_similarities_for_candidates.reshape(-1, 1)).flatten()

        hybrid_scores_initial = np.copy(normalized_mf_similarities) # Initialize
        item_weights_list = [0.0] * len(candidate_song_ids) # Initialize for candidate songs
        
        for idx_in_candidate, song_id in enumerate(candidate_song_ids):
            if exclude_seed and song_id == seed_item_id:
                hybrid_scores_initial[idx_in_candidate] = -1.0 # Mark as excluded
                item_weights_list[idx_in_candidate] = 0.0 # Dummy weight for excluded item
                continue
            
            # Ensure required mappings for item weight calculation are present.
            # Removed self.song_to_language from this check as language is handled by filtering
            if not all([self.song_to_tier, self.song_to_pop, self.song_to_fam, self.song_to_genres, self.tier_weights]): 
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
                    None, None, # Language weights are no longer passed here
                    self.tier_weights
                ) 
                
                # Apply additional boost based on genre match
                item_weight *= (1 + genre_match_boost * 0.5) # Example: 50% boost if genre matches

            item_weights_list[idx_in_candidate] = item_weight # Assign calculated item weight

        valid_item_weights = [w for i, w in enumerate(item_weights_list) if hybrid_scores_initial[i] >= 0]
        if valid_item_weights and (max(valid_item_weights) - min(valid_item_weights)) > 1e-10:
            self.item_weight_scaler.fit(np.array(valid_item_weights).reshape(-1, 1))
            normalized_item_weights = self.item_weight_scaler.transform(np.array(item_weights_list).reshape(-1, 1)).flatten()
        else:
            normalized_item_weights = np.zeros_like(item_weights_list)

        hybrid_scores_final_values = np.copy(hybrid_scores_initial) # Start with current state (incl. exclusions)

        for idx_in_candidate in range(len(candidate_song_ids)):
            if hybrid_scores_final_values[idx_in_candidate] == -1.0: # If excluded, skip
                continue
            
            final_score = (
                normalized_mf_similarities[idx_in_candidate] * self.mf_component_weight +
                normalized_item_weights[idx_in_candidate] * self.item_weight_component_weight
            )
            hybrid_scores_final_values[idx_in_candidate] = final_score
            
        hybrid_scores = hybrid_scores_final_values
        
        valid_scores_mask = hybrid_scores >= 0
        valid_scores = hybrid_scores[valid_scores_mask]

        if valid_scores.size == 0 or (valid_scores.max() - valid_scores.min()) < 1e-10:
            logger.warning(f"Final scores for seed {seed_item_id} are uniform or empty after exclusion. Cannot normalize meaningfully. Returning empty recommendations.")
            return pd.DataFrame(columns=['seed_item_id', 'song_id', 'score']) 
        
        # Ensure normalization only happens if there's a range to normalize over
        if valid_scores.max() - valid_scores.min() > 1e-10:
            hybrid_scores[valid_scores_mask] = (valid_scores - valid_scores.min()) / (valid_scores.max() - valid_scores.min() + 1e-10)
        else:
            # If all valid scores are identical, set them to a neutral value (e.g., 0.5)
            hybrid_scores[valid_scores_mask] = 0.5
        
        ranked_song_indices_in_scores = np.argsort(-hybrid_scores)
        
        top_n_songs_info = []
        top_candidate_count = max(n * 5, 20)

        for i_in_scores in ranked_song_indices_in_scores:
            song_id = candidate_song_ids[i_in_scores] # Get actual song_id from the candidate list
            if hybrid_scores[i_in_scores] < 0:
                continue
            
            top_n_songs_info.append({
                'seed_item_id': seed_item_id,
                'song_id': song_id,
                'score': hybrid_scores[i_in_scores]
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
