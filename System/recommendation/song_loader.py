"""Loader for song-related data in the music recommendation system."""

import os
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class SongLoader:
    """Loader for song-related data including features, metadata, and artist info."""
    
    def __init__(self, data_dir: str):
        """Initialize the song loader.
        
        Args:
            data_dir: Directory containing song data files

        Raises:
            FileNotFoundError: If data_dir does not exist
        """
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory {data_dir} does not exist")
        self.data_dir = data_dir
        self.song_features: Optional[pd.DataFrame] = None
        self.song_metadata: Optional[pd.DataFrame] = None
        self.artist_identification: Optional[pd.DataFrame] = None
        
    def load_artist_identification(self) -> pd.DataFrame:
        """Load artist identification data with popularity metrics.
        
        Returns:
            DataFrame containing artist identification data
        """
        if self.artist_identification is not None:
            logger.info("Returning cached artist identification")
            return self.artist_identification
        
        artist_id_path = os.path.join(self.data_dir, "artist_metadata.csv")
        
        if not os.path.exists(artist_id_path):
            logger.warning("No artist identification file found at %s; returning empty DataFrame", artist_id_path)
            self.artist_identification = pd.DataFrame()
            return self.artist_identification
        
        logger.info("Loading artist identification from %s", artist_id_path)
        try:
            self.artist_identification = pd.read_csv(artist_id_path)
            
            # Check for artist name column (allow 'artist_name', 'artist', or 'name')
            artist_name_col = next((col for col in ['artist_name', 'artist', 'name'] if col in self.artist_identification.columns), None)
            required_cols = ['artist_id', 'external_popularity', 'external_familiarity']
            if artist_name_col:
                required_cols.append(artist_name_col)
            
            if not all(col in self.artist_identification.columns for col in required_cols):
                logger.warning("Missing required columns in artist identification: %s; returning empty DataFrame", required_cols)
                self.artist_identification = pd.DataFrame()
                return self.artist_identification
            
            # Rename artist name column to 'artist_name' if necessary
            if artist_name_col and artist_name_col != 'artist_name':
                logger.info("Renaming artist name column from '%s' to 'artist_name'", artist_name_col)
                self.artist_identification = self.artist_identification.rename(columns={artist_name_col: 'artist_name'})
            
            logger.info("Artist identification loaded with columns: %s", list(self.artist_identification.columns))
            return self.artist_identification
        except Exception as e:
            logger.error("Failed to load artist identification: %s", str(e))
            self.artist_identification = pd.DataFrame()
            return self.artist_identification
        
    def load_song_features(self) -> pd.DataFrame:
        """Load and preprocess song features.
        
        Returns:
            DataFrame containing song audio features
        """
        features_path = os.path.join(self.data_dir, "song_audio_features.csv")
        
        if not os.path.exists(features_path):
            logger.error("No song features found at %s", features_path)
            self.song_features = pd.DataFrame()
            return self.song_features
            
        logger.info("Loading song features from %s", features_path)
        try:
            self.song_features = pd.read_csv(features_path)
            self.song_features=self.song_features.drop(columns=["danceability","energy"])
            self._handle_missing_features()
            return self.song_features
        except Exception as e:
            logger.error("Failed to load song features: %s", str(e))
            self.song_features = pd.DataFrame()
            return self.song_features
        
    def _handle_missing_features(self) -> None:
        """Handle missing values in song features."""
        # This method is not the cause of the non-determinism, as confirmed.
        # Keeping it for completeness, but assuming no missing values in practice.
        if self.song_features is None or self.song_features.empty:
            logger.warning("No song features to handle missing values for")
            return
        nan_counts = self.song_features.isna().sum()
        nan_columns = nan_counts[nan_counts > 0]
        for col in nan_columns.index:
            if pd.api.types.is_numeric_dtype(self.song_features[col]):
                # IMPORTANT: If you do have missing values here, ensure this logic
                # is sound for your data. For now, assuming no missing features.
                self.song_features[col] = self.song_features[col].fillna(self.song_features[col].mean())
        logger.info("Handled missing values in %d columns", len(nan_columns))
    
    def load_song_metadata(self) -> pd.DataFrame:
        """Load and process song metadata.
        
        Returns:
            DataFrame containing song metadata with artist information
        """
        metadata_path = os.path.join(self.data_dir, "song_metadata.csv")
        
        if not os.path.exists(metadata_path):
            logger.error("No song metadata found at %s", metadata_path)
            self.song_metadata = pd.DataFrame()
            return self.song_metadata
            
        logger.info("Loading song metadata from %s", metadata_path)
        try:
            self.song_metadata = pd.read_csv(metadata_path)
            
            # Ensure consistent song_id column
            id_cols = ['song_id', 'track_id', 'id']
            id_col = next((col for col in id_cols if col in self.song_metadata.columns), self.song_metadata.columns[0])
            if id_col != 'song_id':
                logger.info("Renaming song ID column from '%s' to 'song_id'", id_col)
                self.song_metadata = self.song_metadata.rename(columns={id_col: 'song_id'})
            
            # Handle artist_name column
            artist_name_col = next((col for col in ['artist_name', 'artist', 'name'] if col in self.song_metadata.columns), None)
            if artist_name_col and artist_name_col != 'artist_name':
                logger.info("Renaming artist column from '%s' to 'artist_name'", artist_name_col)
                self.song_metadata = self.song_metadata.rename(columns={artist_name_col: 'artist_name'})
            has_artist_name = artist_name_col is not None
            
            # Merge with artist_identification for external_popularity and external_familiarity
            merge_cols = ['artist_id', 'external_popularity', 'external_familiarity']
            if not has_artist_name and self.artist_identification is not None and 'artist_name' in self.artist_identification.columns:
                logger.info("No artist_name in song_metadata; including artist_name from artist_identification")
                merge_cols.append('artist_name')
            
            if self.artist_identification is not None:
                logger.info("Merging artist metadata for columns: %s", merge_cols)
                pre_merge_rows = len(self.song_metadata)
                self.song_metadata = self.song_metadata.merge(
                    self.artist_identification[merge_cols],
                    on='artist_id',
                    how='left'
                )
                logger.info("Merge complete: %d rows before, %d rows after", pre_merge_rows, len(self.song_metadata))
            else:
                logger.warning("No artist identification data available for merge")
            
            # Set defaults for missing columns
            if 'artist_name' not in self.song_metadata.columns:
                logger.warning("artist_name missing; setting to 'Unknown'")
                self.song_metadata['artist_name'] = 'Unknown'
            if 'external_popularity' not in self.song_metadata.columns:
                logger.warning("external_popularity missing; setting to 0.0")
                self.song_metadata['external_popularity'] = 0.0
            if 'external_familiarity' not in self.song_metadata.columns:
                logger.warning("external_familiarity missing; setting to 0.0")
                self.song_metadata['external_familiarity'] = 0.0
            if 'artist_tier' not in self.song_metadata.columns:
                logger.warning("artist_tier column missing; setting to 'Unknown'")
                self.song_metadata['artist_tier'] = 'Unknown'
            
            # Process genre tags
            genre_columns = [col for col in self.song_metadata.columns if col.startswith('genre_')]
            top_genre_columns = [col for col in self.song_metadata.columns if col.startswith('top_genre_')]
            
            if top_genre_columns:
                logger.info("Processing top_genre_ columns with deterministic tie-breaking.")
                # Sort columns alphabetically to ensure deterministic tie-breaking
                sorted_top_genre_columns = sorted(top_genre_columns)
                self.song_metadata['top_genre'] = self.song_metadata[sorted_top_genre_columns].idxmax(axis=1).str.replace('top_genre_', '')
            elif genre_columns:
                logger.info("Processing genre_ columns as fallback with deterministic tie-breaking.")
                # Sort columns alphabetically to ensure deterministic tie-breaking
                sorted_genre_columns = sorted(genre_columns)
                self.song_metadata['top_genre'] = self.song_metadata[sorted_genre_columns].idxmax(axis=1).str.replace('genre_', '')
            else:
                logger.warning("No genre_ or top_genre_ columns found in song_metadata; setting to 'Unknown'.")
                self.song_metadata['top_genre'] = 'Unknown'
            
            # Retain genre_ columns for feature use
            if genre_columns:
                logger.info("Retaining %d genre_ columns for features", len(genre_columns))
            else:
                logger.warning("No genre_ columns available for features")
            
            # Process language columns
            language_columns = [col for col in self.song_metadata.columns if col.startswith('language_')]
            if language_columns:
                logger.info("Processing language tags")
                self.song_metadata['language'] = self.song_metadata[language_columns].idxmax(axis=1).str.replace('language_', '').str.capitalize()
            else:
                logger.warning("No language columns found in song_metadata")
                self.song_metadata['language'] = 'Unknown'
            
            # Log final columns
            logger.info("Song metadata loaded with %d songs", len(self.song_metadata))
            return self.song_metadata
        except Exception as e:
            logger.error("Failed to load song metadata: %s", str(e))
            self.song_metadata = pd.DataFrame()
            return self.song_metadata
    
    def get_song_features(self) -> pd.DataFrame:
        """Get song features DataFrame.
        
        Returns:
            DataFrame containing song audio features
        """
        if self.song_features is None:
            self.load_song_features()
        return self.song_features
    
    def get_song_metadata(self) -> pd.DataFrame:
        """Get song metadata DataFrame.
        
        Returns:
            DataFrame containing song metadata
        """
        if self.song_metadata is None:
            self.load_song_metadata()
        return self.song_metadata
    
    def get_artist_identification(self) -> pd.DataFrame:
        """Get artist identification DataFrame.
        
        Returns:
            DataFrame containing artist identification data
        """
        if self.artist_identification is None:
            self.load_artist_identification()
        return self.artist_identification