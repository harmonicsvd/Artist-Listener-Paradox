import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from System.recommendation.base import RecommenderBase
from System.recommendation.content_based import ContentBasedRecommender
from System.recommendation.matrix_factorization import MatrixFactorizationRecommender

logger = logging.getLogger(__name__)

class HybridRecommender(RecommenderBase):
    """Hybrid recommender combining content-based and matrix factorization approaches."""

    def __init__(
        self,
        content_recommender: ContentBasedRecommender,
        mf_recommender: MatrixFactorizationRecommender,
        tier_weights: Optional[Dict[str, float]] = None,
        content_weight: float = 0.5,
        mf_weight: float = 0.5,
        name: str = "HybridContentMF",
        song_to_artist: Optional[Dict[str, str]] = None # Added for exposure tracking
    ):
        super().__init__(name=name)
        self.content_recommender = content_recommender
        self.mf_recommender = mf_recommender
        self.tier_weights = tier_weights
        self.content_weight = content_weight
        self.mf_weight = mf_weight
        self.song_metadata = None
        self.is_trained = False
        self.song_to_artist = song_to_artist # Store song_to_artist for exposure analysis

        # Ensure weights sum to 1.0
        total_weight = self.content_weight + self.mf_weight
        if total_weight == 0:
            logger.warning("Content and MF weights sum to zero. Defaulting to 0.5 for both.")
            self.content_weight = 0.5
            self.mf_weight = 0.5
        else:
            self.content_weight /= total_weight
            self.mf_weight /= total_weight
        logger.info(f"Hybrid weights initialized: Content={self.content_weight:.2f}, MF={self.mf_weight:.2f}")

    def train(
        self,
        user_interactions: pd.DataFrame,
        song_features: pd.DataFrame = None,
        song_metadata: pd.DataFrame = None,
        artist_identification: Optional[pd.DataFrame] = None,
        testing_mode: bool = False,
        cache_dir: Optional[str] = None
    ) -> None:
        """
        Trains both the content-based and matrix factorization recommenders.
        """
        if testing_mode:
            logger.info(f"Training {self.name} recommender (delegating to sub-recommenders)")

        self.song_metadata = song_metadata
        self.is_trained = False # Reset training status

        # Ensure song_to_artist is set for the hybrid recommender
        if self.song_to_artist is None and song_metadata is not None:
            self.song_to_artist = dict(zip(song_metadata['song_id'], song_metadata.get('artist_name', pd.Series(index=song_metadata['song_id'], dtype=str))))
            logger.info("HybridRecommender: song_to_artist mapping initialized from song_metadata.")

        # Train Content-Based Recommender
        logger.info("Training Content-Based component...")
        self.content_recommender.train(
            user_interactions=user_interactions,
            song_features=song_features,
            song_metadata=song_metadata,
            artist_identification=artist_identification,
            testing_mode=testing_mode,
            cache_dir=cache_dir
        )

        # Train Matrix Factorization Recommender
        logger.info("Training Matrix Factorization component...")
        self.mf_recommender.train(
            user_interactions=user_interactions,
            song_features=song_features, # MF might not directly use song_features, but passing for consistency
            song_metadata=song_metadata,
            artist_identification=artist_identification,
            testing_mode=testing_mode,
            cache_dir=cache_dir
        )

        if self.content_recommender.is_trained and self.mf_recommender.is_trained:
            self.is_trained = True
            logger.info(f"{self.name} training complete (both components trained).")
        else:
            logger.warning(f"{self.name} training incomplete: Content-Based trained: {self.content_recommender.is_trained}, MF trained: {self.mf_recommender.is_trained}")

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_listened: bool = True,
        include_metadata: bool = True,
        testing_mode: bool = False,
        user_interactions: Optional[pd.DataFrame] = None # Pass interactions for on-the-fly profile building
    ) -> pd.DataFrame:
        """
        Generates hybrid recommendations for a user by combining content-based and MF results.
        """
        if not self.is_trained:
            logger.error(f"Recommender '{self.name}' is not trained.")
            return pd.DataFrame(columns=['user_id', 'song_id', 'score'])

        if testing_mode:
            logger.debug(f"Generating {n} hybrid recommendations for user {user_id}")

        # Get recommendations from Content-Based Recommender
        content_recs = self.content_recommender.recommend(
            user_id=user_id,
            n=n * 2, # Get more to allow for filtering and merging
            exclude_listened=exclude_listened,
            include_metadata=include_metadata,
            testing_mode=testing_mode,
            user_interactions=user_interactions # Pass for on-the-fly profile
        )
        if content_recs.empty:
            logger.warning("Content-Based recommender returned no recommendations.")

        # Get recommendations from Matrix Factorization Recommender
        mf_recs = self.mf_recommender.recommend(
            user_id=user_id,
            n=n * 2, # Get more to allow for filtering and merging
            exclude_listened=exclude_listened,
            include_metadata=include_metadata,
            testing_mode=testing_mode,
            user_interactions=user_interactions # Pass for on-the-fly profile
        )
        if mf_recs.empty:
            logger.warning("Matrix Factorization recommender returned no recommendations.")

        if content_recs.empty and mf_recs.empty:
            logger.warning("Both content-based and MF recommenders returned empty results for user %s.", user_id)
            return pd.DataFrame(columns=['user_id', 'song_id', 'score'])

        # Prepare for merging: ensure consistent column names and index
        content_recs = content_recs[['user_id', 'song_id', 'score']].set_index('song_id')
        mf_recs = mf_recs[['user_id', 'song_id', 'score']].set_index('song_id')

        # Combine scores using outer merge to keep all unique songs
        combined_recs = content_recs.merge(
            mf_recs,
            left_index=True,
            right_index=True,
            how='outer',
            suffixes=('_content', '_mf')
        )

        # Fill NaN scores with 0 (or a neutral value) for songs present in one but not the other
        combined_recs['score_content'] = combined_recs['score_content'].fillna(0)
        combined_recs['score_mf'] = combined_recs['score_mf'].fillna(0)

        # Calculate hybrid score
        combined_recs['score'] = (
            self.content_weight * combined_recs['score_content'] +
            self.mf_weight * combined_recs['score_mf']
        )

        # Reset index to make 'song_id' a column again
        combined_recs = combined_recs.reset_index()

        # Ensure 'user_id' column is present after merge (it might be lost if only one recommender had it)
        if 'user_id' not in combined_recs.columns:
            combined_recs['user_id'] = user_id # Assign the current user_id

        # Merge metadata
        if include_metadata and self.song_metadata is not None and not combined_recs.empty:
            desired_columns = ['song_id', 'title', 'artist_name', 'top_genre', 'language', 'artist_id', 'artist_tier']
            available_columns = [col for col in desired_columns if col in self.song_metadata.columns]
            if 'song_id' not in available_columns:
                logger.error("song_id missing from song_metadata; cannot merge metadata for hybrid recommendations.")
            else:
                combined_recs = combined_recs.merge(
                    self.song_metadata[available_columns],
                    on='song_id',
                    how='left'
                )
                # Fill NaNs for safety after merge
                for col in ['title', 'artist_name', 'top_genre', 'language', 'artist_tier', 'artist_id']:
                    if col in combined_recs.columns:
                        if combined_recs[col].dtype == 'object':
                            combined_recs[col] = combined_recs[col].fillna(f'Unknown {col.replace("_", " ").title()}')
                        elif col in ['external_popularity', 'external_familiarity']: # Example for numeric columns
                            combined_recs[col] = combined_recs[col].fillna(0.0)


        # Apply artist exposure logic
        if 'artist_id' in combined_recs.columns and 'artist_tier' in combined_recs.columns:
            max_exposure = {
                'emerging_new': 2, 'emerging_trending': 2, 'rising_established': 2,
                'mid_tier': 2, 'established': 2, 'established_trending': 1, 'established_legacy': 2
            }
            filtered_recs_by_exposure = []
            artist_exposure_tracker = {}

            # Sort by score before applying exposure filter
            for _, row in combined_recs.sort_values('score', ascending=False).iterrows():
                artist_id = str(row['artist_id']) # Ensure artist_id is string for consistent dict keys
                artist_tier = str(row['artist_tier']) # Ensure artist_tier is string
                current_exposure = artist_exposure_tracker.get(artist_id, 0)
                tier_limit = max_exposure.get(artist_tier, 3) # Default limit if tier not found

                if current_exposure < tier_limit:
                    filtered_recs_by_exposure.append(row)
                    artist_exposure_tracker[artist_id] = current_exposure + 1
                if len(filtered_recs_by_exposure) >= n:
                    break

            final_recs = pd.DataFrame(filtered_recs_by_exposure).head(n)
        else:
            logger.warning("Missing artist_id or artist_tier in recommendations; skipping exposure logic.")
            final_recs = combined_recs.sort_values('score', ascending=False).head(n)

        if testing_mode:
            logger.debug(f"Generated {len(final_recs)} hybrid recommendations for user {user_id}.")
            logger.debug(f"Final recommendations columns: {list(final_recs.columns)}")
        return final_recs

    def recommend_similar_items(
        self,
        seed_item_id: str,
        n: int = 10,
        exclude_seed: bool = True,
        include_metadata: bool = True,
        user_id: Optional[str] = None, # User ID for personalization in similar items
        testing_mode: bool = False
    ) -> pd.DataFrame:
        """
        Generates hybrid similar item recommendations by combining content-based and MF results.
        """
        if not self.is_trained:
            logger.error(f"Recommender '{self.name}' is not trained.")
            return pd.DataFrame(columns=['seed_item_id', 'song_id', 'score'])

        if testing_mode:
            logger.debug(f"Generating {n} hybrid similar items for seed {seed_item_id}")

        # Get similar items from Content-Based Recommender
        content_similar_items = self.content_recommender.recommend_similar_items(
            seed_item_id=seed_item_id,
            n=n * 2, # Get more to allow for filtering and merging
            exclude_seed=exclude_seed,
            include_metadata=include_metadata,
            user_id=user_id, # Pass user_id for personalized similar items
            testing_mode=testing_mode
        )
        if content_similar_items.empty:
            logger.warning("Content-Based similar items recommender returned no recommendations.")


        # Get similar items from Matrix Factorization Recommender
        mf_similar_items = self.mf_recommender.recommend_similar_items(
            seed_item_id=seed_item_id,
            n=n * 2, # Get more to allow for filtering and merging
            exclude_seed=exclude_seed,
            include_metadata=include_metadata,
            user_id=user_id, # Pass user_id for personalized similar items
            testing_mode=testing_mode
        )
        if mf_similar_items.empty:
            logger.warning("Matrix Factorization similar items recommender returned no recommendations.")

        if content_similar_items.empty and mf_similar_items.empty:
            logger.warning("Both content-based and MF similar items recommenders returned empty results for seed %s.", seed_item_id)
            return pd.DataFrame(columns=['seed_item_id', 'song_id', 'score'])

        # Prepare for merging: ensure consistent column names and index
        # Ensure 'seed_item_id' column is handled correctly for merging
        content_similar_items = content_similar_items[['song_id', 'score']].set_index('song_id')
        mf_similar_items = mf_similar_items[['song_id', 'score']].set_index('song_id')

        # Combine scores using outer merge to keep all unique songs
        combined_similar_items = content_similar_items.merge(
            mf_similar_items,
            left_index=True,
            right_index=True,
            how='outer',
            suffixes=('_content', '_mf')
        )

        # Fill NaN scores with 0 (or a neutral value)
        combined_similar_items['score_content'] = combined_similar_items['score_content'].fillna(0)
        combined_similar_items['score_mf'] = combined_similar_items['score_mf'].fillna(0)

        # Calculate hybrid score
        combined_similar_items['score'] = (
            self.content_weight * combined_similar_items['score_content'] +
            self.mf_weight * combined_similar_items['score_mf']
        )

        # Reset index to make 'song_id' a column again
        combined_similar_items = combined_similar_items.reset_index()
        combined_similar_items['seed_item_id'] = seed_item_id # Add seed_item_id back

        # Merge metadata
        if include_metadata and self.song_metadata is not None and not combined_similar_items.empty:
            desired_columns = ['song_id', 'title', 'artist_name', 'top_genre', 'language', 'artist_id', 'artist_tier']
            available_columns = [col for col in desired_columns if col in self.song_metadata.columns]
            if 'song_id' not in available_columns:
                logger.error("song_id missing from song_metadata; cannot merge metadata for hybrid similar items.")
            else:
                combined_similar_items = combined_similar_items.merge(
                    self.song_metadata[available_columns],
                    on='song_id',
                    how='left'
                )
                # Fill NaNs for safety after merge
                for col in ['title', 'artist_name', 'top_genre', 'language', 'artist_tier', 'artist_id']:
                    if col in combined_similar_items.columns:
                        if combined_similar_items[col].dtype == 'object':
                            combined_similar_items[col] = combined_similar_items[col].fillna(f'Unknown {col.replace("_", " ").title()}')
                        elif col in ['external_popularity', 'external_familiarity']:
                            combined_similar_items[col] = combined_similar_items[col].fillna(0.0)

        # Apply artist exposure logic
        if 'artist_id' in combined_similar_items.columns and 'artist_tier' in combined_similar_items.columns:
            max_exposure = {
                'emerging_new': 2, 'emerging_trending': 2, 'rising_established': 2,
                'mid_tier': 2, 'established': 2, 'established_trending': 1, 'established_legacy': 2
            }
            filtered_recs_by_exposure = []
            artist_exposure_tracker = {}

            # Sort by score before applying exposure filter
            for _, row in combined_similar_items.sort_values('score', ascending=False).iterrows():
                artist_id = str(row['artist_id'])
                artist_tier = str(row['artist_tier'])
                current_exposure = artist_exposure_tracker.get(artist_id, 0)
                tier_limit = max_exposure.get(artist_tier, 3)

                if current_exposure < tier_limit:
                    filtered_recs_by_exposure.append(row)
                    artist_exposure_tracker[artist_id] = current_exposure + 1
                if len(filtered_recs_by_exposure) >= n:
                    break

            final_recs = pd.DataFrame(filtered_recs_by_exposure).head(n)
        else:
            logger.warning("Missing artist_id or artist_tier in similar items recommendations; skipping exposure logic.")
            final_recs = combined_similar_items.sort_values('score', ascending=False).head(n)

        if testing_mode:
            logger.debug(f"Generated {len(final_recs)} hybrid similar items for seed {seed_item_id}.")
            logger.debug(f"Final similar items columns: {list(final_recs.columns)}")
        return final_recs

