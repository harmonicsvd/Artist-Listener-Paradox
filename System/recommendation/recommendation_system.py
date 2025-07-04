from typing import Dict, Optional, Tuple
import pandas as pd
from System.recommendation.base import RecommenderBase
from System.recommendation.dataloading import RecommendationDataManager
from System.recommendation.utils.exposure import track_exposure, analyze_exposure_distribution
from System.recommendation.utils.recommendation_generation import get_recommendations, get_similar_items
from System.recommendation.utils.recommendation_analysis import compare_recommendation_approaches
import logging

logger = logging.getLogger(__name__)

class RecommendationSystem:
    """Main class orchestrating the recommendation system."""
    
    def __init__(self, data_dir: str):
        """Initialize the recommendation system.
        
        Args:
            data_dir: Directory containing all data files
        """
        self.data_manager = RecommendationDataManager(data_dir)
        self.recommenders: Dict[str, RecommenderBase] = {}
        self.artist_exposure: Dict[str, int] = {}
        
    def add_recommender(self, recommender: RecommenderBase) -> None:
        """Add a recommender to the system.
        
        Args:
            recommender: Recommender instance
        """
        self.recommenders[recommender.name] = recommender
        logger.info("Added recommender: %s", recommender.name)
        
    def load_data(self, max_users: Optional[int] = None) -> None:
        """Load all required data using the data manager.
        
        Args:
            max_users: Optional limit on number of users
        """
        logger.info("Loading all data")
        try:
            self.data_manager.load_all_data(max_users=max_users)
        except ValueError as e:
            logger.error("Failed to load data: %s", str(e))
            raise
        
        # Validate loaded data
        if self.data_manager.user_interactions is None or self.data_manager.user_interactions.empty:
            logger.error("User interactions are empty or not loaded")
        if self.data_manager.song_features is None or self.data_manager.song_features.empty:
            logger.error("Song features are empty or not loaded")
        if self.data_manager.song_metadata is None or self.data_manager.song_metadata.empty:
            logger.error("Song metadata are empty or not loaded")
        if self.data_manager.artist_identification is None or self.data_manager.artist_identification.empty:
            logger.warning("Artist identification is empty or not loaded")
        
    def train_recommender(self, recommender_name: str) -> None:
        """Train a specific recommender using cached data.
        
        Args:
            recommender_name: Name of the recommender to train
            
        Raises:
            KeyError: If recommender_name is not found
            ValueError: If critical data is not loaded
        """
        if recommender_name not in self.recommenders:
            raise KeyError(f"Recommender {recommender_name} not found")
        
        # Check critical data
        missing_data = []
        if self.data_manager.user_interactions is None or self.data_manager.user_interactions.empty:
            missing_data.append("user_interactions")
        if self.data_manager.song_features is None or self.data_manager.song_features.empty:
            missing_data.append("song_features")
        if self.data_manager.song_metadata is None or self.data_manager.song_metadata.empty:
            missing_data.append("song_metadata")
        
        if missing_data:
            raise ValueError(f"Critical data missing: {', '.join(missing_data)}. Call load_data first.")
        
        # Warn if artist_identification is missing
        if self.data_manager.artist_identification is None or self.data_manager.artist_identification.empty:
            logger.warning("Training without artist identification data")
        
        logger.info("Training recommender: %s", recommender_name)
        recommender = self.recommenders[recommender_name]
        recommender.train(
            user_interactions=self.data_manager.user_interactions,
            song_features=self.data_manager.song_features,
            song_metadata=self.data_manager.song_metadata,
            artist_identification=self.data_manager.artist_identification
        )
        
    def get_recommendations(
        self,
        user_id: str,
        recommender_name: str,
        n: int = 10,
        include_user_profile: bool = True,
        verbose: bool = True,
        testing_mode: bool = False
    ) -> Tuple[pd.DataFrame, Dict]:
        """Generate recommendations for a user.
        
        Args:
            user_id: Target user ID
            recommender_name: Name of the recommender
            n: Number of recommendations
            include_user_profile: Include user profile in output
            verbose: Print progress messages
            testing_mode: Enable detailed logging
            
        Returns:
            Tuple of (recommendations DataFrame, user profile dict)
            
        Raises:
            KeyError: If recommender_name is not found
        """
        if recommender_name not in self.recommenders:
            raise KeyError(f"Recommender {recommender_name} not found")
        
        return get_recommendations(
            system=self,
            user_id=user_id,
            recommender=self.recommenders[recommender_name],
            n=n,
            include_user_profile=include_user_profile,
            verbose=verbose,
            testing_mode=testing_mode
        )
        
    def get_similar_items(
        self,
        seed_item_id: str,
        recommender_name: str,
        n: int = 10,
        verbose: bool = True,
        
    ) -> pd.DataFrame:
        """Generate similar item recommendations.
        
        Args:
            seed_item_id: Seed item ID
            recommender_name: Name of the recommender
            n: Number of recommendations
            verbose: Print progress messages
            
        Returns:
            Recommendations DataFrame
            
        Raises:
            KeyError: If recommender_name is not found
        """
        if recommender_name not in self.recommenders:
            raise KeyError(f"Recommender {recommender_name} not found")
        
        return get_similar_items(
            system=self,
            seed_item_id=seed_item_id,
            recommender=self.recommenders[recommender_name],
            n=n,
            verbose=verbose
        )
        
    def track_exposure(self, recommendations: pd.DataFrame, song_to_artist: Dict[str, str]) -> None:
        """Track artist exposure from recommendations.
        
        Args:
            recommendations: DataFrame with recommendations
            song_to_artist: Mapping from song IDs to artist names
        """
        if recommendations.empty:
            logger.warning("No recommendations to track exposure for")
            return
        
        exposure_df = track_exposure(
            recommendations=recommendations,
            song_to_artist=song_to_artist,
            artist_exposure=self.artist_exposure
        )
        for _, row in exposure_df.iterrows():
            self.artist_exposure[row['artist_name']] = row['exposure_count']
        #logger.info("Updated exposure for %d artists", len(exposure_df))
        
    def analyze_exposure_distribution(self, song_metadata: Optional[pd.DataFrame] = None) -> Dict:
        """Analyze exposure across artist tiers.
        
        Args:
            song_metadata: DataFrame containing song metadata with artist tier information
        
        Returns:
            Dict with exposure analysis including exposure_df, tier_diversity, and gini_coefficient
        """
        return analyze_exposure_distribution(
            artist_exposure=self.artist_exposure,
            song_metadata=song_metadata if song_metadata is not None else self.data_manager.song_metadata
        )
        
    def compare_recommendation_approaches(
        self,
        user_id: str,
        seed_item_id: str,
        recommender_name: str,
        n: int = 10,
        testing_mode: bool = False,
        suppress_language_distribution: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compare user-item and item-item recommendations.
        
        Args:
            user_id: Target user ID
            seed_item_id: Seed item ID
            recommender_name: Name of the recommender
            n: Number of recommendations
            testing_mode: Enable detailed logging
            suppress_language_distribution: Suppress language distribution output
            
        Returns:
            Tuple of (user-item recommendations, item-item recommendations)
            
        Raises:
            KeyError: If recommender_name is not found
        """
        if recommender_name not in self.recommenders:
            raise KeyError(f"Recommender {recommender_name} not found")
        
        return compare_recommendation_approaches(
            system=self,
            user_id=user_id,
            seed_item_id=seed_item_id,
            recommender_name=recommender_name,
            n=n,
            testing_mode=testing_mode,
            suppress_language_distribution=suppress_language_distribution
        )