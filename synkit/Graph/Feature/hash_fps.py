import networkx as nx
import hashlib
from typing import Optional, Any


class HashFPs:
    def __init__(
        self, graph: nx.Graph, numBits: int = 256, hash_alg: str = "sha256"
    ) -> None:
        """
        Initialize the HashFPs class with a graph and configuration settings.

        Parameters:
        - graph (nx.Graph): The graph to be fingerprinted.
        - numBits (int): Number of bits in the output binary hash. Default is 256 bits.
        - hash_alg (str): The hash algorithm to use, such as 'sha256' or 'sha512'.

        Raises:
        - ValueError: If `numBits` is non-positive or if `hash_alg` is not supported
        by hashlib.
        """
        self.graph = graph
        self.numBits = numBits
        self.hash_alg = hash_alg
        self.validate_parameters()

    def validate_parameters(self) -> None:
        """Validate the initial parameters for errors."""
        if self.numBits <= 0:
            raise ValueError("Number of bits must be positive")
        if not hasattr(hashlib, self.hash_alg):
            raise ValueError(f"Unsupported hash algorithm: {self.hash_alg}")

    def hash_fps(
        self,
        start_node: Optional[int] = None,
        end_node: Optional[int] = None,
        max_path_length: Optional[int] = None,
    ) -> str:
        """
        Generate a binary hash fingerprint of the graph based on its paths and cycles.

        Parameters:
        - start_node (Optional[int]): The starting node index for path detection.
        - end_node (Optional[int]): The ending node index for path detection.
        - max_path_length (Optional[int]): The maximum length for paths to be considered.

        Returns:
        - str: A binary string representing the truncated hash of the graph's structural
        features.
        """
        hash_object = self.initialize_hash()
        features = self.extract_features(start_node, end_node, max_path_length)
        full_hash_binary = self.finalize_hash(hash_object, features)
        return full_hash_binary

    def initialize_hash(self) -> Any:
        """Initialize and return the hash object based on the specified algorithm."""
        return getattr(hashlib, self.hash_alg)()

    def extract_features(
        self,
        start_node: Optional[int],
        end_node: Optional[int],
        max_path_length: Optional[int],
    ) -> str:
        """
        Extract features from the graph based on paths and cycles.

        Parameters:
        - start_node (Optional[int]): The starting node for path detection.
        - end_node (Optional[int]): The ending node for path detection.
        - max_path_length (Optional[int]): Cutoff for path length during detection.

        Returns:
        - str: A string of concatenated feature values.
        """
        cycles = list(nx.simple_cycles(self.graph))
        paths = []
        if start_node is not None and end_node is not None:
            paths = list(
                nx.all_simple_paths(
                    self.graph,
                    source=start_node,
                    target=end_node,
                    cutoff=max_path_length,
                )
            )
        features = [len(c) for c in cycles] + [len(p) for p in paths]
        return "".join(map(str, features))

    def finalize_hash(self, hash_object: Any, features: str) -> str:
        """
        Finalize the hash using the features extracted and return the hash as a binary
        string.

        Parameters:
        - hash_object (Any): The hash object.
        - features (str): Concatenated string of graph features.

        Returns:
        - str: The final binary string of the hash, truncated or extended to `numBits`.
        """
        hash_object.update(features.encode())
        full_hash_binary = bin(int(hash_object.hexdigest(), 16))[2:]
        if len(full_hash_binary) < self.numBits:
            full_hash_binary += self.iterative_deepening(
                hash_object, self.numBits - len(full_hash_binary)
            )
        return full_hash_binary[: self.numBits]

    def iterative_deepening(self, hash_object: Any, remaining_bits: int) -> str:
        """
        Extend hash length using iterative hashing until the desired bit length is
        achieved.

        Parameters:
        - hash_object (hashlib._Hash): The hash object for iterative deepening.
        - remaining_bits (int): Number of bits needed to reach `numBits`.

        Returns:
        - str: Additional binary data to achieve the desired hash length.
        """
        additional_data = ""
        while (
            len(additional_data) * 4 < remaining_bits
        ):  # Each hex digit represents 4 bits
            hash_object.update(additional_data.encode())
            additional_data += hash_object.hexdigest()
        return bin(int(additional_data, 16))[2:][:remaining_bits]
