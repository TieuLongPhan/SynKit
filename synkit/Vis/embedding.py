from typing import Any, Dict, Optional
import numpy as np
from sklearn.manifold import TSNE
from joblib import Memory


class Embedding:
    def __init__(
        self,
        cache_dir: str = "./cachedir",
        verbose: int = 0,
        custom_tsne_params: Optional[Dict] = None,
    ) -> None:
        """
        Initialize the Embedding class with options for caching directory, verbosity, and custom t-SNE parameters.

        Parameters:
            cache_dir (str): Directory where cached results are stored.
            verbose (int): Verbosity level for the memory object.
            custom_tsne_params (Dict, optional): Custom default parameters for t-SNE computations.
        """
        self.memory = Memory(cache_dir, verbose=verbose)
        self.default_tsne_params = {
            "n_components": 2,
            "perplexity": 30,
            "learning_rate": 200,
            "max_iter": 1000,
            "random_state": 42,
        }
        if custom_tsne_params:
            self.default_tsne_params.update(custom_tsne_params)
        self.tsne_params = self.default_tsne_params.copy()

    def set_tsne_params(self, **params) -> None:
        """
        Sets parameters for t-SNE computations.

        Parameters:
            **params: Arbitrary number of parameters for t-SNE.
        """
        self.tsne_params.update(params)

    def reset_tsne_params(self) -> None:
        """
        Resets t-SNE parameters to default values.
        """
        self.tsne_params = self.default_tsne_params.copy()

    def _compute_tsne(self, X: np.ndarray) -> np.ndarray:
        """
        Direct computation of the t-SNE embedding with the current parameters.

        Parameters:
            X (np.ndarray): High-dimensional data points.

        Returns:
            np.ndarray: The 2-dimensional t-SNE embedding of the data.
        """
        tsne = TSNE(**self.tsne_params)
        return tsne.fit_transform(X)

    def compute_tsne(self, X: np.ndarray, cache: bool = True) -> np.ndarray:
        """
        Computes or retrieves the t-SNE embedding from cache.

        Parameters:
            X (np.ndarray): High-dimensional data points.
            cache (bool): Determines whether to use caching for the computation.

        Returns:
            np.ndarray: The 2-dimensional t-SNE embedding of the data.
        """
        if cache:
            return self.cache(X)
        else:
            return self._compute_tsne(X)

    @property
    def cache(self) -> Any:
        """
        Decorator for caching the compute_tsne function.

        Returns:
            Callable: Cached function.
        """
        return self.memory.cache(self._compute_tsne)

    def clear_cache(self) -> None:
        """
        Clears the cache directory.
        """
        self.memory.clear()
