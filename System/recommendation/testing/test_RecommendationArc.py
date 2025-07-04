import unittest
import pandas as pd
import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from System.recommendation.recommendation_system import RecommendationSystem


class TestExposureTracker(unittest.TestCase):
    def setUp(self):
        # Initialize a RecommendationSystem instance
        self.rec_system = RecommendationSystem(data_dir="/dummy/path")

    def test_exposure_tracker_with_valid_data(self):
        # Mock recommendations DataFrame
        recommendations = pd.DataFrame({
            'song_id': ['song1', 'song2', 'song3', 'song1', 'song2']
        })

        # Mock song-to-artist mapping
        song_to_artist_map = {
            'song1': 'artist1',
            'song2': 'artist2',
            'song3': 'artist3'
        }

        # Call the exposure_tracker method
        exposure_df = self.rec_system.exposure_tracker(recommendations, song_to_artist_map)

        # Expected result
        expected_data = {
            'artist_id': ['artist1', 'artist2', 'artist3'],
            'exposure_count': [2, 2, 1]
        }
        expected_df = pd.DataFrame(expected_data)
        print("\nOutput of test_name:")
        print(exposure_df)
    
        # Assert the result matches the expected DataFrame
        pd.testing.assert_frame_equal(exposure_df.sort_values(by='artist_id').reset_index(drop=True),
                                      expected_df.sort_values(by='artist_id').reset_index(drop=True))

    def test_exposure_tracker_with_empty_recommendations(self):
        # Mock empty recommendations DataFrame
        recommendations = pd.DataFrame(columns=['song_id'])

        # Mock song-to-artist mapping
        song_to_artist_map = {
            'song1': 'artist1',
            'song2': 'artist2',
            'song3': 'artist3'
        }

        # Call the exposure_tracker method
        exposure_df = self.rec_system.exposure_tracker(recommendations, song_to_artist_map)

        # Expected result
        expected_df = pd.DataFrame(columns=['artist_id', 'exposure_count'])
        print("\nOutput of test_name:")
        print(exposure_df)

        # Assert the result matches the expected DataFrame
        pd.testing.assert_frame_equal(exposure_df, expected_df)

    def test_exposure_tracker_with_missing_song_ids(self):
        # Mock recommendations DataFrame with a song ID not in the mapping
        recommendations = pd.DataFrame({
            'song_id': ['song1', 'song4', 'song2']
        })

        # Mock song-to-artist mapping
        song_to_artist_map = {
            'song1': 'artist1',
            'song2': 'artist2',
            'song3': 'artist3'
        }

        # Call the exposure_tracker method
        exposure_df = self.rec_system.exposure_tracker(recommendations, song_to_artist_map)

        # Expected result
        expected_data = {
            'artist_id': ['artist1', 'artist2'],
            'exposure_count': [1, 1]
        }
        expected_df = pd.DataFrame(expected_data)
        print("\nOutput of test_name:")
        print(exposure_df)

        # Assert the result matches the expected DataFrame
        pd.testing.assert_frame_equal(exposure_df.sort_values(by='artist_id').reset_index(drop=True),
                                      expected_df.sort_values(by='artist_id').reset_index(drop=True))

if __name__ == '__main__':
    unittest.main()                