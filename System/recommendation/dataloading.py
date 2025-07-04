"""Data loading orchestrator for the music recommendation system."""

import os
from typing import Optional
import pandas as pd
import logging
from System.recommendation.song_loader import SongLoader
from System.recommendation.user_loader import UserLoader
import numpy as np

logger = logging.getLogger(__name__)

class RecommendationDataManager:
    """Manager for all recommendation data loading and alignment."""
    
    def __init__(self, data_dir: str):
        """Initialize the data manager.
        
        Args:
            data_dir: Directory containing all recommendation data files

        Raises:
            FileNotFoundError: If data_dir does not exist
        """
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory {data_dir} does not exist")
        self.data_dir = data_dir
        self.song_loader = SongLoader(data_dir)
        self.user_loader = UserLoader(data_dir)
        self.user_interactions: Optional[pd.DataFrame] = None
        self.song_features: Optional[pd.DataFrame] = None
        self.song_metadata: Optional[pd.DataFrame] = None
        self.artist_identification: Optional[pd.DataFrame] = None
        
    def load_all_data(self, max_users: Optional[int] = None) -> None:
        """Load all required data for recommendations and cache it.

        Args:
            max_users: Optional limit on number of users to load

        Raises:
            ValueError: If critical data fails to load
        """
        logger.info("Loading song features, metadata, and artist data")
        
        # Load song features
        try:
            self.song_features = self.song_loader.load_song_features()
            if self.song_features is None or self.song_features.empty:
                logger.error("Song features are empty or not loaded")
            else:
                logger.info("Loaded %d song features", len(self.song_features))
        except Exception as e:
            logger.error("Failed to load song features: %s", str(e))
            self.song_features = pd.DataFrame()
        
        # Load artist identification
        try:
            self.artist_identification = self.song_loader.load_artist_identification()
            if self.artist_identification is None or self.artist_identification.empty:
                logger.warning("Artist identification is empty or not loaded")
            else:
                logger.info("Loaded %d artist identification records", len(self.artist_identification))
        except Exception as e:
            logger.error("Failed to load artist identification: %s", str(e))
            self.artist_identification = pd.DataFrame()
        
        # Load song metadata (after artist_identification to ensure merge)
        try:
            self.song_metadata = self.song_loader.load_song_metadata()
            if self.song_metadata is None or self.song_metadata.empty:
                logger.error("Song metadata are empty or not loaded")
            else:
                logger.info("Loaded %d song metadata records", len(self.song_metadata))
        except Exception as e:
            logger.error("Failed to load song metadata: %s", str(e))
            self.song_metadata = pd.DataFrame()
        
        logger.info("Loading user interactions and preferences")
        try:
            self.user_loader.load_interactions()
            self.user_interactions = self.user_loader.interactions  # Cache interactions
            if self.user_interactions is None or self.user_interactions.empty:
                logger.error("User interactions are empty or not loaded")
            else:
                logger.info("Loaded %d user interactions", len(self.user_interactions))
        except Exception as e:
            logger.error("Failed to load user interactions: %s", str(e))
            self.user_interactions = pd.DataFrame()
        
        # Filter users if max_users is specified
        if max_users is not None and self.user_interactions is not None and not self.user_interactions.empty:
            unique_users = self.user_interactions['user_id'].unique()
            if len(unique_users) > max_users:
                selected_users = np.random.choice(unique_users, max_users, replace=False)
                self.user_interactions = self.user_interactions[self.user_interactions['user_id'].isin(selected_users)]
                self.user_loader.interactions = self.user_interactions
                logger.info("Filtered interactions to %d users", max_users)
        
        try:
            self.user_loader.load_preferences()
            if self.user_loader.preferences is None or (isinstance(self.user_loader.preferences, dict) and not self.user_loader.preferences):
                logger.warning("User preferences are empty or not loaded")
            elif isinstance(self.user_loader.preferences, dict):
                logger.info("Loaded preferences for %d users", len(self.user_loader.preferences))
            elif isinstance(self.user_loader.preferences, pd.DataFrame):
                logger.info("Loaded preferences for %d users", len(self.user_loader.preferences))
            else:
                logger.warning("User preferences loaded in unsupported format: %s", type(self.user_loader.preferences))
        except Exception as e:
            logger.error("Failed to load user preferences: %s", str(e))
        
        # Log summary
        logger.info("Loaded data: %d interactions, %d songs, %d artist records",
                   len(self.user_interactions) if self.user_interactions is not None else 0,
                   len(self.song_metadata) if self.song_metadata is not None else 0,
                   len(self.artist_identification) if self.artist_identification is not None else 0)
        
        # Raise error if critical data is missing
        if any(df is None or df.empty for df in [self.song_features, self.song_metadata, self.user_interactions]):
            raise ValueError("Critical data (song features, metadata, or interactions) failed to load")