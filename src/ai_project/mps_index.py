import torch

class MPSVectorIndex:
    """Lightweight vector index using PyTorch MPS"""
    
    def __init__(self, device='mps'):
        self.device = device if torch.backends.mps.is_available() else 'cpu'
        self.vectors = None
        self.dim = None
    
    def add(self, embeddings):
        """Add vectors to index"""
        self.vectors = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        self.dim = self.vectors.shape[1]
    
    def search(self, query, k=5):
        """Find k nearest neighbors using L2 distance"""
        if self.vectors is None:
            return [], []
        
        query_tensor = torch.tensor(query, dtype=torch.float32, device=self.device)
        
        # Compute L2 distances
        distances = torch.cdist(query_tensor, self.vectors, p=2)
        
        # Get top k smallest distances
        k = min(k, self.vectors.shape[0])
        values, indices = torch.topk(distances, k, largest=False, dim=1)
        
        return values.cpu().numpy(), indices.cpu().numpy()
