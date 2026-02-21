import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ObjectiveLossCalculator:
    """Calculate Objective Loss combining Listener and Artist Satisfaction."""
    
    def __init__(self, recommender_name=None):
        """Initialize with recommender-specific weights.
        
        Args:
            recommender_name (str): Name of the recommender ('ContentBased', 'CollaborativeFiltering', 'MatrixFactorization', 'HybridContentMF').
        """
        # Define weights for each recommender and k value
        self.weight_configs = {
    'ContentBased': {
        2: {
            'Genre Precision': 0.3500,
            'NDCG': 0.1000,
            'Emerging Artist Hit Rate': 0.1700,
            'Emerging Artist Exposure Index': 0.2300,
            'Coverage': 0.0700,
            'Diversity': 0.0800,   # total = 1.0000
        },
        5: {
            'Genre Precision': 0.4000,
            'NDCG': 0.0800,
            'Emerging Artist Hit Rate': 0.1400,
            'Emerging Artist Exposure Index': 0.2800,
            'Coverage': 0.0500,
            'Diversity': 0.0500,   # total = 1.0000
        }
    },
    'CollaborativeFiltering': {
        2: {
            'Genre Precision': 0.2900,
            'NDCG': 0.1200,
            'Emerging Artist Hit Rate': 0.1300,
            'Emerging Artist Exposure Index': 0.2100,
            'Coverage': 0.1000,
            'Diversity': 0.1500,   # total = 1.0000 (was 0.95)
        },
        5: {
            'Genre Precision': 0.3500,
            'NDCG': 0.1000,
            'Emerging Artist Hit Rate': 0.1200,
            'Emerging Artist Exposure Index': 0.2300,
            'Coverage': 0.0800,
            'Diversity': 0.1200,   # total = 1.0000 (was 0.98)
        }
    },
    'MatrixFactorization': {
        2: {
            'Genre Precision': 0.2900,
            'NDCG': 0.1200,
            'Emerging Artist Hit Rate': 0.1300,
            'Emerging Artist Exposure Index': 0.2100,
            'Coverage': 0.1000,
            'Diversity': 0.1500,   # total = 1.0000 (was 0.95)
        },
        5: {
            'Genre Precision': 0.3500,
            'NDCG': 0.1000,
            'Emerging Artist Hit Rate': 0.1200,
            'Emerging Artist Exposure Index': 0.2300,
            'Coverage': 0.0800,
            'Diversity': 0.1200,   # total = 1.0000 (was 0.98)
        }
    },
    'HybridContentMF': {
        2: {
            'Genre Precision': 0.3500,
            'NDCG': 0.1000,
            'Emerging Artist Hit Rate': 0.1700,
            'Emerging Artist Exposure Index': 0.2300,
            'Coverage': 0.0700,
            'Diversity': 0.0800,   # total = 1.0000
        },
        5: {
            'Genre Precision': 0.4000,
            'NDCG': 0.0800,
            'Emerging Artist Hit Rate': 0.1400,
            'Emerging Artist Exposure Index': 0.2800,
            'Coverage': 0.0500,
            'Diversity': 0.0500,   # total = 1.0000
        }
    }
}

        
        # Select weights based on recommender_name
        if recommender_name in self.weight_configs:
            self.weight_configs = self.weight_configs[recommender_name]
            logger.info(f"Loaded weight configurations for {recommender_name}")
        else:
            # Default to MatrixFactorization weights if recommender_name is not specified or invalid
            self.weight_configs = self.weight_configs['MatrixFactorization']
            logger.warning(f"No recommender_name specified or invalid ('{recommender_name}'); using default MatrixFactorization weights")
        
        # Verify weights sum to 1.0 for each k (listener satisfaction weights)
        for k in [2, 5]:
            if k in self.weight_configs:
                weight_sum = sum(self.weight_configs[k].values())
                # Adjust for floating point inaccuracies
                if not np.isclose(weight_sum, 1.0, rtol=1e-4):
                    logger.error(f"Listener Satisfaction weights sum to {weight_sum:.4f} for k={k}; expected 1.0. Re-normalizing.")
                    # Re-normalize weights if they don't sum to 1.0
                    factor = 1.0 / weight_sum
                    for metric, weight in self.weight_configs[k].items():
                        self.weight_configs[k][metric] = weight * factor
                    logger.info(f"Weights for k={k} re-normalized. New sum: {sum(self.weight_configs[k].values()):.4f}")
                else:
                    logger.debug(f"Weight sum for {recommender_name or 'default'}, k={k}: {weight_sum:.4f}")
        
        # Artist Satisfaction weights (sum to 1.0) - these remain unchanged as they directly address exposure
        self.as_weights = {
            'tier_diversity': 0.45,
            'gini': 0.55
        }
        
        # Objective Loss weights (sum to 1.0)
        self.loss_weights = {'listener': 0.5, 'artist': 0.5}
        
    def compute_listener_satisfaction(self, metrics, k):
        """Compute LS(u) for users based on evaluation metrics.
        
        Args:
            metrics (dict): Metrics dictionary with keys like 'Precision', 'Genre Precision', etc.
            k (int): Number of recommendations (e.g., 2, 5).
        
        Returns:
            float: Average LS(u) across users.
        """
        try:
            ls = 0.0
            ls_weights = self.weight_configs.get(k, self.weight_configs[2])  # Default to k=2 if k not found
            for metric, weight in ls_weights.items():
                value = float(metrics.get(metric, {}).get(k, 0.0))
                # Special handling for Emerging Artist Exposure Index: penalize deviation from 1
                if metric == 'Emerging Artist Exposure Index':
                    # LS component for Exposure Index should be higher when closer to 1.0
                    # We penalize deviation, so (1 - abs(value - 1.0)) results in 1.0 if value is 1.0, 0.0 if value is 0.0 or 2.0
                    ls += weight * (1 - abs(value - 1.0))
                else:
                    ls += weight * value
            logger.info(f"Listener Satisfaction for k={k}: {ls:.4f}")
            return max(0, min(1, ls))  # Clamp to [0, 1]
        except Exception as e:
            logger.error(f"Error computing Listener Satisfaction for k={k}: {e}")
            return 0.0
    
    def compute_artist_satisfaction(self, tier_diversity, gini):
        """Compute AS using precomputed Tier Diversity and Gini.
        
        Args:
            tier_diversity (float): Precomputed Tier Diversity score.
            gini (float): Precomputed Gini coefficient.
        
        Returns:
            float: Artist Satisfaction.
        """
        try:
            # Gini should be lower for better artist satisfaction, so we use (1 - gini)
            as_score = (self.as_weights['tier_diversity'] * float(tier_diversity)) + (self.as_weights['gini'] * (1 - float(gini)))
            logger.info(f"Artist Satisfaction: {as_score:.4f} (TD={tier_diversity:.4f}, Gini={gini:.4f})")
            return max(0, min(1, as_score))  # Clamp to [0, 1]
        except Exception as e:
            logger.error(f"Error computing Artist Satisfaction: {e}")
            return 0.0
    
    def compute_objective_loss(self, metrics, k_values, tier_diversity_by_k, gini_by_k):
        """
        Compute Objective Loss for given k values using precomputed metrics.
        Returns loss, listener satisfaction, and artist satisfaction for each k.
        
        Args:
            metrics (dict): Evaluation metrics.
            k_values (list): List of k values (e.g., [2, 5]).
            tier_diversity_by_k (dict): Tier Diversity scores per k.
            gini_by_k (dict): Gini coefficients per k.
        
        Returns:
            tuple: (dict: loss values per k,
                    dict: listener satisfaction scores per k,
                    dict: artist satisfaction scores per k)
        """
        loss_results = {}
        ls_results = {}
        as_results = {}

        for k in k_values:
            ls = self.compute_listener_satisfaction(metrics, k)
            as_score = self.compute_artist_satisfaction(
                tier_diversity_by_k.get(k, 0.0),
                gini_by_k.get(k, 0.0)
            )
            # Objective loss is calculated as 1 - satisfaction (since we want to minimize loss)
            loss = (self.loss_weights['listener'] * (1 - ls)) + (self.loss_weights['artist'] * (1 - as_score))
            
            loss_results[k] = float(loss)
            ls_results[k] = float(ls)
            as_results[k] = float(as_score)

            logger.info(f"Objective Loss for k={k}: {loss:.4f} (LS={ls:.4f}, AS={as_score:.4f})")
        
        return loss_results, ls_results, as_results

def main(metrics, k_values, tier_diversity_by_k, gini_by_k, recommender_name=None):
    """Main function to compute Objective Loss (example usage)."""
    calculator = ObjectiveLossCalculator(recommender_name=recommender_name)
    loss, ls, as_score = calculator.compute_objective_loss(metrics, k_values, tier_diversity_by_k, gini_by_k)
    return loss, ls, as_score

if __name__ == "__main__":
    # Example usage
    example_metrics = {
        'Precision': {2: 0.2263, 5: 0.1647},
        'Genre Precision': {2: 0.5026, 5: 0.2072},
        'Language Precision': {2: 0.5001, 5: 0.2069},
        'Recall': {2: 0.1509, 5: 0.2745},
        'NDCG': {2: 0.2355, 5: 0.2575},
        'Hit Rate': {2: 0.4111, 5: 0.6221},
        'Emerging Artist Hit Rate': {2: 0.4478, 5: 0.6629},
        'Diversity': {2: 0.6819, 5: 0.7193},
        'Novelty': {2: 0.5216, 5: 0.5049},
        'Emerging Artist Exposure Index': {2: 1.2, 5: 1.3}
    }
    example_tier_diversity = {2: 0.74, 5: 0.73}
    example_gini = {2: 0.7574, 5: 0.7454}
    k_values = [2, 5]
    loss, ls, as_score = main(example_metrics, k_values, example_tier_diversity, example_gini, recommender_name='MatrixFactorization')
    print(f"Objective Loss: {loss}")
    print(f"Listener Satisfaction: {ls}")
    print(f"Artist Satisfaction: {as_score}")
