"""Loader for user-related data in the music recommendation system."""

import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class UserLoader:
    """Loader for user-related data including interactions and preferences."""
    
    def __init__(self, data_dir: str):
        """Initialize the user loader.
        
        Args:
            data_dir: Directory containing user data files

        Raises:
            FileNotFoundError: If data_dir does not exist
        """
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory {data_dir} does not exist")
        self.data_dir = data_dir
        self.interactions = None
        self.preferences = None
        
    def load_interactions(self) -> None:
        """Load and standardize user-song interactions."""
        interactions_path = os.path.join(self.data_dir, "user_interactions_with_language_augmented.csv")
        
        if not os.path.exists(interactions_path):
            raise FileNotFoundError(f"No interactions found at {interactions_path}")
            
        logger.info("Loading user interactions from %s", interactions_path)
        self.interactions = pd.read_csv(interactions_path)
        self._standardize_interaction_columns()
    
    def _standardize_interaction_columns(self) -> None:
        """Standardize column names in interactions DataFrame."""
        column_mapping = {
            'user': 'user_id',
            'song': 'song_id',
            'track_id': 'song_id',
            'item_id': 'song_id',
            'item': 'song_id',
            'count': 'play_count'
        }
        self.interactions = self.interactions.rename(columns={
            k: v for k, v in column_mapping.items() if k in self.interactions.columns
        })
        logger.info("Standardized interaction columns: %s", self.interactions.columns.tolist())
    
    def load_preferences(self) -> None:
        """Load and parse user preferences."""
        preferences_path = os.path.join(self.data_dir, "user_preferences.csv")
        
        if os.path.exists(preferences_path):
            logger.info("Loading user preferences from %s", preferences_path)
            preferences_df = pd.read_csv(preferences_path)
            self._parse_preferences(preferences_df)
        else:
            logger.warning("No user preferences file found at %s", preferences_path)
            self.preferences = {}
    
    def _parse_preferences(self, df: pd.DataFrame) -> None:
        """Convert preferences DataFrame to nested dictionary."""
        self.preferences = {}
        for _, row in df.iterrows():
            user_id = row['user_id']
            self.preferences[user_id] = {col: row[col] for col in df.columns if col != 'user_id'}
        logger.info("Parsed preferences for %d users", len(self.preferences))
    
    def get_interactions(self) -> pd.DataFrame:
        """Get user-song interactions DataFrame."""
        if self.interactions is None:
            self.load_interactions()
        return self.interactions
    
    def get_preferences(self) -> dict:
        """Get user preferences dictionary."""
        if self.preferences is None:
            self.load_preferences()
        return self.preferences