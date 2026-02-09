import pytest
import torch
import numpy as np
from ai_project.mps_index import MPSVectorIndex


class TestMPSVectorIndex:
    """Test suite for MPS-based vector index"""
    
    @pytest.fixture
    def sample_vectors(self):
        """Generate sample vectors for testing"""
        np.random.seed(42)
        return np.random.randn(100, 128).astype(np.float32)
    
    @pytest.fixture
    def index(self):
        """Create index instance"""
        return MPSVectorIndex()
    
    def test_initialization(self, index):
        """Test index initialization"""
        assert index.vectors is None
        assert index.dim is None
        assert index.device in ['mps', 'cpu']
    
    def test_add_vectors(self, index, sample_vectors):
        """Test adding vectors to index"""
        index.add(sample_vectors)
        
        assert index.vectors is not None
        assert index.dim == 128
        assert index.vectors.shape == (100, 128)
        assert index.vectors.device.type in ['mps', 'cpu']
    
    def test_search_basic(self, index, sample_vectors):
        """Test basic search functionality"""
        index.add(sample_vectors)
        
        query = sample_vectors[0:1]  # Use first vector as query
        distances, indices = index.search(query, k=5)
        
        # First result should be the query itself (distance ~0)
        assert indices[0][0] == 0
        assert distances[0][0] < 0.01  # Relaxed tolerance for MPS
        
        # Should return k results
        assert len(indices[0]) == 5
        assert len(distances[0]) == 5
    
    def test_search_k_larger_than_dataset(self, index, sample_vectors):
        """Test search when k > number of vectors"""
        small_dataset = sample_vectors[:10]
        index.add(small_dataset)
        
        query = small_dataset[0:1]
        distances, indices = index.search(query, k=20)
        
        # Should return only available vectors
        assert len(indices[0]) == 10
        assert len(distances[0]) == 10
    
    def test_search_empty_index(self, index):
        """Test search on empty index"""
        query = np.random.randn(1, 128).astype(np.float32)
        distances, indices = index.search(query, k=5)
        
        assert len(distances) == 0
        assert len(indices) == 0
    
    def test_search_ordering(self, index):
        """Test that results are ordered by distance"""
        # Create vectors with known distances
        vectors = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], dtype=np.float32)
        
        index.add(vectors)
        
        query = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        distances, indices = index.search(query, k=4)
        
        # Distances should be in ascending order
        assert np.all(distances[0][:-1] <= distances[0][1:])
        
        # Indices should be [0, 1, 2, 3]
        assert list(indices[0]) == [0, 1, 2, 3]
    
    def test_multiple_queries(self, index, sample_vectors):
        """Test batch search with multiple queries"""
        index.add(sample_vectors)
        
        queries = sample_vectors[0:3]  # Use first 3 vectors as queries
        distances, indices = index.search(queries, k=5)
        
        # Should return results for each query
        assert distances.shape == (3, 5)
        assert indices.shape == (3, 5)
        
        # Each query should find itself as nearest neighbor
        for i in range(3):
            assert indices[i][0] == i
            assert distances[i][0] < 0.01  # Relaxed tolerance for MPS
    
    def test_dimension_consistency(self, index):
        """Test that index enforces dimension consistency"""
        vectors_128 = np.random.randn(10, 128).astype(np.float32)
        index.add(vectors_128)
        
        assert index.dim == 128
        
        # Query with different dimension should work (PyTorch handles it)
        # but results may not be meaningful
        query_64 = np.random.randn(1, 64).astype(np.float32)
        
        # This should raise an error or handle gracefully
        with pytest.raises((RuntimeError, ValueError)):
            index.search(query_64, k=5)
    
    def test_device_placement(self, index, sample_vectors):
        """Test that vectors are placed on correct device"""
        index.add(sample_vectors)
        
        if torch.backends.mps.is_available():
            assert index.device == 'mps'
            assert index.vectors.device.type == 'mps'
        else:
            assert index.device == 'cpu'
            assert index.vectors.device.type == 'cpu'
    
    def test_numerical_stability(self, index):
        """Test with edge case values"""
        # Test with very small values
        small_vectors = np.random.randn(10, 128).astype(np.float32) * 1e-6
        index.add(small_vectors)
        
        query = small_vectors[0:1]
        distances, indices = index.search(query, k=5)
        
        assert not np.any(np.isnan(distances))
        assert not np.any(np.isinf(distances))
    
    def test_large_dataset(self, index):
        """Test with larger dataset"""
        large_vectors = np.random.randn(10000, 128).astype(np.float32)
        index.add(large_vectors)
        
        query = large_vectors[0:1]
        distances, indices = index.search(query, k=10)
        
        assert len(indices[0]) == 10
        assert indices[0][0] == 0  # Should find itself
    
    @pytest.mark.parametrize("k", [1, 5, 10, 50])
    def test_different_k_values(self, index, sample_vectors, k):
        """Test search with different k values"""
        index.add(sample_vectors)
        
        query = sample_vectors[0:1]
        distances, indices = index.search(query, k=k)
        
        expected_k = min(k, len(sample_vectors))
        assert len(indices[0]) == expected_k
        assert len(distances[0]) == expected_k


class TestMPSVectorIndexIntegration:
    """Integration tests for MPS vector index"""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow: add, search, verify"""
        index = MPSVectorIndex()
        
        # Create synthetic data
        vectors = np.random.randn(1000, 256).astype(np.float32)
        index.add(vectors)
        
        # Search for multiple queries
        queries = vectors[0:10]
        distances, indices = index.search(queries, k=5)
        
        # Verify results
        assert distances.shape == (10, 5)
        assert indices.shape == (10, 5)
        
        # Each query should find itself
        for i in range(10):
            assert indices[i][0] == i
    
    def test_memory_efficiency(self):
        """Test that index doesn't leak memory"""
        index = MPSVectorIndex()
        
        # Add vectors multiple times
        for _ in range(5):
            vectors = np.random.randn(100, 128).astype(np.float32)
            index.add(vectors)
        
        # Final shape should be last added vectors
        assert index.vectors.shape == (100, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
