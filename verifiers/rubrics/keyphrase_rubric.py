from typing import List, Dict, Any, Tuple
import numpy as np
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric

class KeyphraseRubric(Rubric):
    def __init__(self):
        super().__init__()
        self.parser = XMLParser(fields=["keywords"])
        self.reward_funcs = [
            self.keyphrase_match_reward_func,
            self.parser.get_xml_reward_func(),
            self.parser.get_format_reward_func()
        ]
        
        # Initialize embedding model on first use to avoid loading during import
        self._embedding_model = None
    
    @property
    def embedding_model(self):
        """Lazy-load the embedding model when first needed"""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.logger.info("Initializing sentence transformer model for keyphrase similarity")
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
            except ImportError:
                self.logger.warning("sentence-transformers not installed. Using exact match fallback.")
                self._embedding_model = False
        return self._embedding_model
    
    def hungarian_similarity(self, 
                            predicted_embeddings: np.ndarray, 
                            ground_truth_embeddings: np.ndarray, 
                            penalty_cost: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Computes the optimal matching between predicted and ground truth embeddings using the Hungarian algorithm.
        
        Args:
            predicted_embeddings: Embeddings of generated keyphrases
            ground_truth_embeddings: Embeddings of reference keyphrases
            penalty_cost: Cost for unmatched items
            
        Returns:
            A tuple of (row_indices, col_indices, matched_similarities, average_similarity)
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            self.logger.warning("scipy or sklearn not installed. Cannot compute Hungarian similarity.")
            return np.array([]), np.array([]), np.array([]), 0.0
            
        # Compute the pairwise cosine similarity matrix
        similarity_matrix = cosine_similarity(predicted_embeddings, ground_truth_embeddings)
        cost_matrix = -similarity_matrix  # Higher similarity → lower cost
        
        # Pad cost matrix to handle uneven numbers of keyphrases
        N, M = cost_matrix.shape
        size = max(N, M)
        padded_cost = np.full((size, size), penalty_cost, dtype=float)
        padded_cost[:N, :M] = cost_matrix
        
        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(padded_cost)
        
        # Filter out dummy matches (those exceeding original dimensions)
        valid_matches = [(i, j, similarity_matrix[i, j])
                         for i, j in zip(row_ind, col_ind) if i < N and j < M]
        
        matched_similarities = np.array([sim for (_, _, sim) in valid_matches])
        avg_similarity = np.mean(matched_similarities) if matched_similarities.size > 0 else 0.0
        
        return row_ind, col_ind, matched_similarities, avg_similarity
    
    def keyphrase_match_reward_func(self, samples: List[Dict[str, Any]]) -> List[float]:
        """
        Evaluates semantic similarity between generated and reference keyphrases.
        Uses embedding-based similarity with Hungarian algorithm for optimal matching.
        Falls back to exact match if dependencies aren't available.
        """
        rewards = []
        
        for sample in samples:
            generated = sample.get("generation", "").strip()
            reference = sample.get("reference", "").strip()
            
            # Parse keyphrases
            generated_phrases = [p.strip() for p in generated.split(",") if p.strip()]
            reference_phrases = [p.strip() for p in reference.split(",") if p.strip()]
            
            # No phrases to compare
            if not generated_phrases or not reference_phrases:
                rewards.append(0.0)
                continue
                
            # Try to use embedding similarity if possible
            if self.embedding_model:
                try:
                    # Encode all phrases
                    generated_embeddings = [
                        self.embedding_model.encode(phrase, normalize_embeddings=True)
                        for phrase in generated_phrases
                    ]
                    reference_embeddings = [
                        self.embedding_model.encode(phrase, normalize_embeddings=True)
                        for phrase in reference_phrases
                    ]
                    
                    # Calculate similarity using Hungarian algorithm
                    _, _, _, avg_similarity = self.hungarian_similarity(
                        np.array(generated_embeddings),
                        np.array(reference_embeddings)
                    )
                    
                    rewards.append(float(avg_similarity))
                    continue
                    
                except Exception as e:
                    self.logger.warning(f"Error computing embedding similarity: {e}. Falling back to exact match.")
            
            # Fallback: simple overlap-based reward (exact matching)
            matches = sum(1 for p in generated_phrases if p in reference_phrases)
            total = max(len(generated_phrases), len(reference_phrases))
            reward = matches / total if total > 0 else 0.0
            rewards.append(reward)
        
        return rewards
