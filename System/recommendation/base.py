"""Abstract base class for all recommender system implementations."""

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd

class RecommenderBase(ABC):
    """Base class defining the interface for all recommendation systems.
    
    Attributes:
        name (str): Identifier for the recommender.
        is_trained (bool): Flag indicating if model has been trained.
        user_ids (List[str]): List of known user IDs.
        item_ids (List[str]): List of known item IDs.
        user_item_matrix (pd.DataFrame): User-item interaction matrix if applicable.
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize the recommender with optional name.
        
        Args:
            name: Optional identifier for the recommender. Defaults to class name if not provided.
        """
        self.name = name or self.__class__.__name__
        self.is_trained = False
        self.user_ids = None
        self.item_ids = None
        self.user_item_matrix = None

    @abstractmethod
    def train(self, user_interactions: pd.DataFrame, *args, **kwargs) -> None:
        """Train the recommendation model.
        
        Args:
            user_interactions: DataFrame containing user-item interactions with columns:
                - user_id: User identifier
                - song_id: Item identifier
                - [other interaction metrics]
            *args: Additional positional arguments for model training
            **kwargs: Additional keyword arguments for model training
        """
        self.user_ids = user_interactions['user_id'].unique().tolist()
        self.item_ids = user_interactions['song_id'].unique().tolist()

    @abstractmethod
    def recommend(self, user_id: str, n: int = 10, exclude_listened: bool = True) -> pd.DataFrame:
        """Generate item recommendations for target user.
        
        Args:
            user_id: Target user identifier
            n: Number of recommendations to return
            exclude_listened: Whether to filter out items the user has already interacted with
            
        Returns:
            DataFrame with recommendations containing at least:
                - user_id: User identifier
                - song_id: Recommended item ID
                - score: Recommendation score
                
        Raises:
            ValueError: If recommender has not been trained
        """
        if not self.is_trained:
            raise ValueError(f"Recommender '{self.name}' is not trained")

    @abstractmethod
    def recommend_similar_items(self, seed_item_id: str, n: int = 10, exclude_seed: bool = True) -> pd.DataFrame:
        """Generate item recommendations based on similarity to a seed item.
        
        Args:
            seed_item_id: ID of the seed item to find similar items for
            n: Number of similar items to return
            exclude_seed: Whether to exclude the seed item from results
            
        Returns:
            DataFrame with recommendations containing at least:
                - seed_item_id: Seed item ID
                - song_id: Recommended item ID
                - score: Similarity score
                
        Raises:
            ValueError: If recommender has not been trained
        """
        if not self.is_trained:
            raise ValueError(f"Recommender '{self.name}' is not trained")