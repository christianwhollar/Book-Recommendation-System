import torch
import torch.nn as nn
import torch.nn.functional as F

class NNHybridFiltering(nn.Module):
    """
    Neural network model for hybrid filtering recommendation.

    Attributes:
        n_users (int): Number of unique users.
        n_isbn (int): Number of unique ISBN values.
        n_bxrating (int): Number of unique Book-Rating values.
        embdim_users (int): Dimension of user embeddings.
        embdim_isbn (int): Dimension of ISBN embeddings.
        embdim_bxrating (int): Dimension of Book-Rating embeddings.
        n_activations (int): Number of activations in the hidden layer.
        rating_range (list[float, float]): Range for scaling predicted ratings.

    Methods:
        __init__(self, n_users, n_isbn, n_bxrating, embdim_users, embdim_isbn, embdim_bxrating, n_activations, rating_range)
            Initialize the NNHybridFiltering model.

        forward(self, X) -> torch.Tensor
            Forward pass of the neural network model.
    """

    def __init__(self, n_users: int, n_isbn: int, n_bxrating: int, embdim_users: int, embdim_isbn: int,
                 embdim_bxrating: int, n_activations: int, rating_range: list[float, float]):
        """
        Initialize the NNHybridFiltering model.

        Args:
            n_users (int): Number of unique users.
            n_isbn (int): Number of unique ISBN values.
            n_bxrating (int): Number of unique Book-Rating values.
            embdim_users (int): Dimension of user embeddings.
            embdim_isbn (int): Dimension of ISBN embeddings.
            embdim_bxrating (int): Dimension of Book-Rating embeddings.
            n_activations (int): Number of activations in the hidden layer.
            rating_range (list[float, float]): Range for scaling predicted ratings.
        """
        super().__init__()
        self.user_embeddings = nn.Embedding(num_embeddings=n_users, embedding_dim=embdim_users)
        self.item_embeddings = nn.Embedding(num_embeddings=n_isbn, embedding_dim=embdim_isbn)
        self.bxrating_embeddings = nn.Embedding(num_embeddings=n_bxrating, embedding_dim=embdim_bxrating)
        self.fc1 = nn.Linear(embdim_users + embdim_isbn + embdim_bxrating, n_activations)
        self.fc2 = nn.Linear(n_activations, 1)
        self.rating_range = rating_range

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network model.

        Args:
            X (torch.Tensor): Input tensor containing user, item, and Book-Rating indices.

        Returns:
            torch.Tensor: Predicted ratings for the input samples.
        """
        # Get embeddings for minibatch
        embedded_users = self.user_embeddings(X[:, 0])
        embedded_isbn = self.item_embeddings(X[:, 1])
        embedded_bxrating = self.bxrating_embeddings(X[:, 2])

        # Concatenate user, item, and bxrating embeddings
        embeddings = torch.cat([embedded_users, embedded_isbn, embedded_bxrating], dim=1)

        # Pass embeddings through network
        preds = self.fc1(embeddings)
        preds = F.relu(preds)
        preds = self.fc2(preds)

        # Scale predicted ratings to target-range [low,high]
        preds = torch.sigmoid(preds) * (self.rating_range[1] - self.rating_range[0]) + self.rating_range[0]
        return preds
