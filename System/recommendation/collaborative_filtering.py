import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Optional, List, Set
import logging
import os
import pickle

from System.recommendation.base import RecommenderBase
from System.recommendation.utils.weights import calculate_item_weights, calculate_genre_weights # Removed calculate_language_weights
from System.recommendation.utils.mappings import create_song_to_tier_mapping, create_genre_mapping, create_language_mapping, create_artist_metric_mappings, compute_related_genres
from tqdm import tqdm

logger = logging.getLogger(__name__)

class CollaborativeFilteringRecommender(RecommenderBase):
    """User-based and item-based collaborative filtering recommender."""
    
    def __init__(
        self,
        tier_weights: Dict[str, float],
        name: str = "CollaborativeFiltering",
        song_to_tier: Optional[Dict[str, str]] = None,
        song_to_pop: Optional[Dict[str, float]] = None,
        song_to_fam: Optional[Dict[str, float]] = None,
        song_to_genres: Optional[Dict[str, List[str]]] = None,
        song_to_language: Optional[Dict[str, str]] = None,
        feature_weights: Optional[Dict[str, float]] = None,
        user_profiles: Optional[Dict[str, Dict]] = None, # Passed at init
        user_extended_genres_cache: Optional[Dict[str, Set[str]]] = None,
        related_genres_map: Optional[Dict[str, Set[str]]] = None,
        song_to_artist: Optional[Dict[str, str]] = None
    ):
        super().__init__(name)
        self.tier_weights = tier_weights
        self.user_item_matrix = None
        self.user_similarities = None
        self.item_similarities = None
        self.user_id_to_index = None
        self.song_id_to_index = None
        self.index_to_song_id = None
        self.index_to_user_id = None 
        self.song_metadata = None
        self.song_to_artist = song_to_artist
        self.item_features = None 
        self.feature_weights = feature_weights or {
            'similarity_component': 3.7, 
            'item_weight_component': 2.5 
        }
        self.item_weight_range_min = 0.0 
        self.item_weight_range_max = 2.0 
        self.user_profiles = user_profiles if user_profiles is not None else {} # Initialize here
        self.user_extended_genres_cache = user_extended_genres_cache if user_extended_genres_cache is not None else {}
        self.song_to_tier = song_to_tier
        self.song_to_pop = song_to_pop
        self.song_to_fam = song_to_fam
        self.song_to_genres = song_to_genres
        self.song_to_language = song_to_language
        self.related_genres_map = related_genres_map 

    def _build_all_user_profiles_for_cf(self, user_interactions: pd.DataFrame) -> Dict[str, Dict]:
        """
        Builds user profiles (genre weights, listened items) for all users
        during CF training. Language weights are no longer calculated here.
        """
        all_user_profiles = {}
        unique_users = user_interactions['user_id'].unique()
        
        logger.info(f"Building initial CF user profiles for {len(unique_users)} users.")
        
        for user_id in tqdm(unique_users, desc="Building CF User Profiles", leave=False, disable=False):
            user_items = user_interactions[user_interactions['user_id'] == user_id]
            if user_items.empty or len(user_items) < 1:
                logger.debug(f"User {user_id} has no interactions or insufficient interactions. Skipping profile creation.")
                continue

            item_ids_listened = user_items['song_id'].tolist()
            play_counts_listened = user_items.get('play_count', pd.Series([1.0] * len(item_ids_listened))).tolist()

            # Ensure essential mappings are available before calculating weights
            if not all([self.song_to_genres, self.related_genres_map]): # Removed self.song_to_language
                logger.warning(f"Essential mappings not fully initialized for user {user_id} profile. Skipping genre weight calculation for this user.")
                # Create a basic profile without weights if mappings are missing
                all_user_profiles[user_id] = {'listened_items': set(item_ids_listened), 'genre_weights': {}}
                continue

            # language_weights removed
            genre_weights = calculate_genre_weights(item_ids_listened, play_counts_listened, self.song_to_genres)
            
            user_genres = {genre for item_id in item_ids_listened for genre in self.song_to_genres.get(item_id, [])}
            related_genres = {rel_genre for genre in user_genres for rel_genre in self.related_genres_map.get(genre, {genre})}
            extended_genres = user_genres.union(related_genres)
            
            all_user_profiles[user_id] = {
                'genre_weights': genre_weights,
                'listened_items': set(item_ids_listened)
            }
            self.user_extended_genres_cache[user_id] = extended_genres # Cache for recommend calls

        logger.info(f"Completed building initial CF user profiles. Generated {len(all_user_profiles)} profiles.")
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
        """
        Train the Collaborative Filtering recommender.
        
        This method attempts to load precomputed matrices and mappings from cache.
        If loading is successful, it uses the cached data. If cache files are missing,
        corrupted, or if a re-computation is implicitly forced by the absence of cache,
        it computes everything from scratch and saves the results to the cache directory.
        """
        logger.info("Training %s recommender", self.name)
        super().train(user_interactions) # Call superclass train to set self.is_trained if needed
        
        self.song_metadata = song_metadata
        self.artist_identification = artist_identification # Ensure this is stored if needed

        if song_metadata is not None:
            if self.song_to_artist is None:
                self.song_to_artist = dict(zip(song_metadata['song_id'], song_metadata.get('artist_name', pd.Series(index=song_metadata['song_id'], dtype=str))))
            if self.related_genres_map is None and self.song_to_genres: 
                logger.info("Computing related genres map for CollaborativeFiltering (during train, if not pre-passed)")
                self.related_genres_map = compute_related_genres(self.song_to_genres)
                logger.info("Computed related genres for %d genres", len(self.related_genres_map))
            elif self.related_genres_map is None:
                logger.warning("Cannot compute related_genres_map: song_to_genres is not set.")
            
            # Ensure song_to_language is populated during train
            if self.song_to_language is None and 'language' in song_metadata.columns:
                self.song_to_language = dict(zip(song_metadata['song_id'], song_metadata['language']))

        if not all([self.song_to_tier, self.song_to_pop, self.song_to_fam, 
                    self.song_to_genres, self.song_to_language, self.related_genres_map]):
            logger.error("CollaborativeFiltering requires essential mappings; missing mappings may cause recommendation errors.")

        # Derive counts from the actual data being processed for cache key consistency
        actual_num_users = user_interactions['user_id'].nunique()
        actual_num_songs = user_interactions['song_id'].nunique()

        # --- Caching Logic ---
        # If cache_dir is provided, attempt to load from cache
        if cache_dir:
            # Construct cache paths with relevant parameters for uniqueness
            cache_matrix_path = os.path.join(cache_dir, f"{self.name.lower()}_user_item_matrix_users{actual_num_users}_songs{actual_num_songs}.npz")
            cache_user_sim_path = os.path.join(cache_dir, f"{self.name.lower()}_user_similarities_users{actual_num_users}_songs{actual_num_songs}.npy")
            cache_item_sim_path = os.path.join(cache_dir, f"{self.name.lower()}_item_similarities_users{actual_num_users}_songs{actual_num_songs}.npy")
            cache_user_idx_map_path = os.path.join(cache_dir, f"{self.name.lower()}_user_id_to_index_users{actual_num_users}_songs{actual_num_songs}.pkl")
            cache_song_idx_map_path = os.path.join(cache_dir, f"{self.name.lower()}_song_id_to_index_users{actual_num_users}_songs{actual_num_songs}.pkl")
            cache_idx_song_map_path = os.path.join(cache_dir, f"{self.name.lower()}_index_to_song_id_users{actual_num_users}_songs{actual_num_songs}.pkl")
            cache_user_profiles_path = os.path.join(cache_dir, f"{self.name.lower()}_user_profiles_users{actual_num_users}.pkl") # New cache path for profiles

            # Check if all cached components exist
            if os.path.exists(cache_matrix_path) and \
               os.path.exists(cache_user_sim_path) and \
               os.path.exists(cache_item_sim_path) and \
               os.path.exists(cache_user_idx_map_path) and \
               os.path.exists(cache_song_idx_map_path) and \
               os.path.exists(cache_idx_song_map_path) and \
               os.path.exists(cache_user_profiles_path): # Check for user profiles cache
                
                logger.info(f"Attempting to load {self.name} matrices and mappings from cache: (users:{actual_num_users}, songs:{actual_num_songs}).")
                try:
                    self.user_item_matrix = load_npz(cache_matrix_path) # Load as sparse matrix
                    self.user_similarities = np.load(cache_user_sim_path)
                    self.item_similarities = np.load(cache_item_sim_path)
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
                    # Fall through to recalculate if loading fails

        logger.info(f"Cache not found or failed to load for {self.name} matrices. Calculating: (users:{actual_num_users}, songs:{actual_num_songs})...")

        # Create mappings from user/song IDs to matrix indices
        unique_users = user_interactions['user_id'].unique()
        unique_songs = user_interactions['song_id'].unique()
        
        self.user_id_to_index = {uid: idx for idx, uid in enumerate(unique_users)}
        self.song_id_to_index = {sid: idx for idx, sid in enumerate(unique_songs)}
        self.index_to_song_id = {idx: sid for sid, idx in self.song_id_to_index.items()}
        self.index_to_user_id = {idx: uid for uid, idx in self.user_id_to_index.items()} 
        self.item_features = np.zeros((len(unique_songs), 1)) 
            
        rows = [self.user_id_to_index[uid] for uid in user_interactions['user_id']]
        cols = [self.song_id_to_index[sid] for sid in user_interactions['song_id']]
        data = user_interactions.get('play_count', pd.Series(np.ones(len(user_interactions)))).values
        
        self.user_item_matrix = csr_matrix((data, (rows, cols)), shape=(len(unique_users), len(unique_songs)))
        
        logger.info("Computing user similarities...")
        self.user_similarities = cosine_similarity(self.user_item_matrix, dense_output=True)
        
        logger.info("Computing item similarities...")
        item_matrix = self.user_item_matrix.T
        self.item_similarities = cosine_similarity(item_matrix, dense_output=True)
        
        logger.info("User-item matrix shape: %s, computed %d user similarities, %d item similarities.", 
                    self.user_item_matrix.shape, len(unique_users), len(unique_songs))

        # Build user profiles (genre weights, listened items) here during train
        self.user_profiles = self._build_all_user_profiles_for_cf(user_interactions)
        logger.debug(f"After CollaborativeFiltering train, user_profiles has {len(self.user_profiles)} entries.")

        self.is_trained = True 
        
        if testing_mode:
            logger.info(f"{self.name} training complete.")

        # Save to cache if cache_dir is provided and training was successful
        if cache_dir and self.is_trained:
            logger.info(f"Saving {self.name} matrices and mappings to cache for U:{actual_num_users}, S:{actual_num_songs}.")
            try:
                os.makedirs(cache_dir, exist_ok=True) 
                save_npz(cache_matrix_path, self.user_item_matrix)
                np.save(cache_user_sim_path, self.user_similarities)
                np.save(cache_item_sim_path, self.item_similarities)
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

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_listened: bool = True, 
        include_metadata: bool = True,
        testing_mode: bool = False,
        user_interactions: Optional[pd.DataFrame] = None # user_interactions is only used for on-the-fly here now
    ) -> pd.DataFrame:
        if not self.is_trained or self.user_item_matrix is None:
            logger.error(f"Recommender '{self.name}' is not trained or has no valid matrix")
            return pd.DataFrame(columns=['user_id', 'song_id', 'score'])
        
        if testing_mode:
            logger.debug("Generating %d recommendations for user %s", n, user_id)
        
        user_profile = self.user_profiles.get(user_id)
        
        # If user_profile is not found in pre-built profiles, attempt on-the-fly building
        if not user_profile and user_interactions is not None:
            logger.debug(f"User profile for {user_id} not found in pre-built cache. Attempting to build on-the-fly.")
            user_items = user_interactions[user_interactions['user_id'] == user_id]
            
            if user_items.empty or len(user_items) < 1: 
                logger.warning(f"User {user_id} has insufficient interactions ({len(user_items)}) to build an on-the-fly profile. Returning empty recommendations.")
                return pd.DataFrame(columns=['user_id', 'song_id', 'score'])

            item_ids_listened = user_items['song_id'].tolist()
            play_counts_listened = user_items.get('play_count', pd.Series([1.0] * len(item_ids_listened))).tolist()
            
            if not all([self.song_to_genres, self.related_genres_map]): # Removed self.song_to_language
                logger.error("Required mappings (song_to_genres, related_genres_map) not initialized for on-the-fly profile creation in CF recommend. Cannot proceed.")
                return pd.DataFrame(columns=['user_id', 'song_id', 'score'])

            # language_weights removed
            genre_weights = calculate_genre_weights(item_ids_listened, play_counts_listened, self.song_to_genres)
            
            user_genres = {genre for item_id in item_ids_listened for genre in self.song_to_genres.get(item_id, [])}
            related_genres = {rel_genre for genre in user_genres for rel_genre in self.related_genres_map.get(genre, {genre})}
            extended_genres = user_genres.union(related_genres)
            
            user_profile = {
                'genre_weights': genre_weights,
                'listened_items': set(item_ids_listened)
            }
            self.user_profiles[user_id] = user_profile 
            self.user_extended_genres_cache[user_id] = extended_genres 
            logger.debug(f"Successfully built on-the-fly profile for user {user_id}.")
        
        if not user_profile: 
            logger.warning("No user profile found or built for user %s; returning empty recommendations", user_id)
            return pd.DataFrame(columns=['user_id', 'song_id', 'score'])
        
        user_idx = self.user_id_to_index.get(user_id)
        if user_idx is None:
            logger.warning(f"User {user_id} not found in trained matrix (user_id_to_index); returning empty recommendations.")
            return pd.DataFrame(columns=['user_id', 'song_id', 'score'])
        
        user_similarity_vector = self.user_similarities[user_idx].flatten()
        
        # Calculate raw collaborative scores for ALL songs first
        raw_collaborative_scores = user_similarity_vector @ self.user_item_matrix

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

        # Extract collaborative scores only for the candidate songs
        collaborative_scores_for_candidates = raw_collaborative_scores[candidate_original_indices]

        if collaborative_scores_for_candidates.size == 0 or (collaborative_scores_for_candidates.max() - collaborative_scores_for_candidates.min()) < 1e-10:
            logger.warning(f"Collaborative scores for user {user_id} (filtered by language) are uniform or empty. Cannot normalize meaningfully. Returning empty recommendations.")
            return pd.DataFrame(columns=['user_id', 'song_id', 'score'])
        
        # Normalize collaborative scores for candidates
        normalized_collaborative_scores = (collaborative_scores_for_candidates - collaborative_scores_for_candidates.min()) / \
                                          (collaborative_scores_for_candidates.max() - collaborative_scores_for_candidates.min() + 1e-10)
        
        scores = []
        listened_songs_indices = {self.song_id_to_index[sid] for sid in user_profile['listened_items'] if sid in self.song_id_to_index}
        
        for idx_in_candidate, song_id in enumerate(candidate_song_ids): # Iterate over candidate song_ids
            original_idx = self.song_id_to_index[song_id] # Get original index for exclusion check
            if exclude_listened and original_idx in listened_songs_indices:
                scores.append(-1.0) 
                continue
            
            if not all([self.song_to_tier, self.song_to_pop, self.song_to_fam, self.song_to_genres, self.tier_weights]):
                logger.warning(f"Required mappings missing for item {song_id} to calculate item weights. Skipping item weighting and using only collaborative score.")
                scores.append(normalized_collaborative_scores[idx_in_candidate] * self.feature_weights['similarity_component'])
                continue

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
            
            min_iw_dynamic = -0.5 
            max_iw_dynamic = 1.5  
            if item_weight < min_iw_dynamic: min_iw_dynamic = item_weight
            if item_weight > max_iw_dynamic: max_iw_dynamic = item_weight

            range_diff = max_iw_dynamic - min_iw_dynamic
            if range_diff <= 1e-10: 
                normalized_item_weight = self.item_weight_range_min 
            else:
                normalized_item_weight = self.item_weight_range_min + \
                                         (item_weight - min_iw_dynamic) / range_diff * \
                                         (self.item_weight_range_max - self.item_weight_range_min)
            
            final_score = (
                normalized_collaborative_scores[idx_in_candidate] * self.feature_weights['similarity_component'] +
                normalized_item_weight * self.feature_weights['item_weight_component']
            )
            scores.append(final_score)
        
        scores = np.array(scores)
        
        valid_scores_mask = scores >= 0
        valid_scores = scores[valid_scores_mask]

        if valid_scores.size == 0 or (valid_scores.max() - valid_scores.min()) < 1e-10:
            logger.warning(f"Final scores for user {user_id} are uniform or empty after exclusion. Cannot normalize meaningfully. Returning empty recommendations.")
            return pd.DataFrame(columns=['user_id', 'song_id', 'score'])
        
        scores[valid_scores_mask] = (valid_scores - valid_scores.min()) / (valid_scores.max() - valid_scores.min() + 1e-10)
        
        ranked_song_indices_in_scores = np.argsort(-scores) # These are indices within the `scores` array (which corresponds to `candidate_song_ids`)
        
        top_n_songs_info = []
        top_candidate_count = max(n * 5, 20) 

        for i_in_scores in ranked_song_indices_in_scores:
            song_id = candidate_song_ids[i_in_scores] # Get actual song_id from the candidate list
            if scores[i_in_scores] < 0: 
                continue
            
            top_n_songs_info.append({
                'user_id': user_id,
                'song_id': song_id,
                'score': scores[i_in_scores]
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
        if not self.is_trained or self.user_item_matrix is None:
            logger.error(f"Recommender '{self.name}' is not trained or has no valid matrix")
            return pd.DataFrame(columns=['seed_item_id', 'song_id', 'score'])
        
        if testing_mode:
            logger.debug("Generating %d similar items for seed %s", n, seed_item_id)
        
        seed_idx = self.song_id_to_index.get(seed_item_id)
        if seed_idx is None:
            logger.warning(f"Seed item {seed_item_id} not found in trained matrix; returning empty recommendations.")
            return pd.DataFrame(columns=['seed_item_id', 'song_id', 'score'])
        
        # Calculate item similarity vector for ALL songs first
        original_item_similarity_vector = self.item_similarities[seed_idx].flatten()

        # Determine target language(s) for similar items
        target_languages = []
        user_listened_items = set()

        if user_id and self.song_to_language:
            # Get user's listened items to determine dominant language
            user_profile = self.user_profiles.get(user_id)
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
        item_similarity_vector_for_candidates = original_item_similarity_vector[candidate_original_indices]
        
        if item_similarity_vector_for_candidates.size == 0 or (item_similarity_vector_for_candidates.max() - item_similarity_vector_for_candidates.min()) < 1e-10:
            logger.warning(f"Item similarity vector for seed item {seed_item_id} (filtered by language) is uniform or empty. Cannot normalize meaningfully.")
            normalized_similarities = np.zeros_like(item_similarity_vector_for_candidates)
        else:
            normalized_similarities = (item_similarity_vector_for_candidates - item_similarity_vector_for_candidates.min()) / \
                                      (item_similarity_vector_for_candidates.max() - item_similarity_vector_for_candidates.min() + 1e-10)
        
        scores = []
        user_profile = self.user_profiles.get(user_id) if user_id else None
        
        extended_genres = set()
        if user_id and user_id in self.user_extended_genres_cache:
            extended_genres = self.user_extended_genres_cache[user_id]
        else:
            seed_genres = set(self.song_to_genres.get(seed_item_id, []))
            for genre in seed_genres:
                extended_genres.update(self.related_genres_map.get(genre, {genre}))
            extended_genres.update(seed_genres)
        
        if testing_mode:
            logger.debug(f"Extended genres size: {len(extended_genres)}")
            
        for idx_in_candidate, song_id in enumerate(candidate_song_ids): # Iterate over candidate song_ids
            if exclude_seed and song_id == seed_item_id:
                scores.append(-1.0) 
                continue
            
            if not all([self.song_to_tier, self.song_to_pop, self.song_to_fam, self.song_to_genres, self.tier_weights]): # Removed self.song_to_language
                logger.warning(f"Required mappings missing for item {song_id}. Skipping item weighting.")
                scores.append(normalized_similarities[idx_in_candidate] * self.feature_weights['similarity_component'])
                continue
            
            item_weight = calculate_item_weights(
                song_id,
                self.song_to_tier,
                self.song_to_pop,
                self.song_to_fam,
                user_profile['genre_weights'] if user_profile else {}, 
                self.song_to_genres,
                None, None, # Language weights are no longer passed here
                self.tier_weights
            )
            
            min_iw_dynamic = -0.5
            max_iw_dynamic = 1.5
            if item_weight < min_iw_dynamic: min_iw_dynamic = item_weight
            if item_weight > max_iw_dynamic: max_iw_dynamic = item_weight

            range_diff = max_iw_dynamic - min_iw_dynamic
            if range_diff <= 1e-10:
                normalized_item_weight = self.item_weight_range_min
            else:
                normalized_item_weight = self.item_weight_range_min + \
                                         (item_weight - min_iw_dynamic) / range_diff * \
                                         (self.item_weight_range_max - self.item_weight_range_min)
            
            final_score = (
                normalized_similarities[idx_in_candidate] * self.feature_weights['similarity_component'] +
                normalized_item_weight * self.feature_weights['item_weight_component']
            )
            scores.append(final_score)
        
        scores = np.array(scores)
        valid_scores_mask = scores >= 0
        valid_scores = scores[valid_scores_mask]

        if valid_scores.size == 0 or (valid_scores.max() - valid_scores.min()) < 1e-10:
            logger.warning(f"Final scores for seed {seed_item_id} are uniform or empty after filtering. Cannot normalize meaningfully. Returning empty recommendations.")
            return pd.DataFrame(columns=['seed_item_id', 'song_id', 'score']) 
        
        scores[valid_scores_mask] = (valid_scores - valid_scores.min()) / (valid_scores.max() - valid_scores.min() + 1e-10)
        
        ranked_song_indices_in_scores = np.argsort(-scores) 
        
        top_n_songs_info = []
        top_candidate_count = max(n * 5, 20)

        for i_in_scores in ranked_song_indices_in_scores:
            song_id = candidate_song_ids[i_in_scores] # Get actual song_id from the candidate list
            if scores[i_in_scores] < 0:
                continue
            
            top_n_songs_info.append({
                'seed_item_id': seed_item_id,
                'song_id': song_id,
                'score': scores[i_in_scores]
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
