import networkx as nx
import hashlib
from typing import Optional, Any


class HashFPs:
    def __init__(
        self, graph: nx.Graph, numBits: int = 256, hash_alg: str = "sha256"
    ) -> None:
        """Initialize the HashFPs class with a graph and configuration
        settings.

        :param graph: The graph to be fingerprinted.
        :type graph: nx.Graph
        :param numBits: Number of bits in the output binary hash. Default is 256 bits.
        :type numBits: int
        :param hash_alg: The hash algorithm to use, such as 'sha256' or 'sha512'.
        :type hash_alg: str

        :raises ValueError: If `numBits` is non-positive or if `hash_alg` is not supported by hashlib.
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
        """Generate a binary hash fingerprint of the graph based on its paths
        and cycles.

        :param start_node: The starting node index for path detection.
        :type start_node: Optional[int]
        :param end_node: The ending node index for path detection.
        :type end_node: Optional[int]
        :param max_path_length: The maximum length for paths to be considered.
        :type max_path_length: Optional[int]

        :return: A binary string representing the truncated hash of the graph's structural features.
        :rtype: str
        """
        hash_object = self.initialize_hash()
        features = self.extract_features(start_node, end_node, max_path_length)
        full_hash_binary = self.finalize_hash(hash_object, features)
        return full_hash_binary

    def initialize_hash(self) -> Any:
        """Initialize and return the hash object based on the specified
        algorithm."""
        return getattr(hashlib, self.hash_alg)()

    def extract_features(
        self,
        start_node: Optional[int],
        end_node: Optional[int],
        max_path_length: Optional[int],
    ) -> str:
        """Extract features from the graph based on paths and cycles.

        :param start_node: The starting node for path detection.
        :type start_node: Optional[int]
        :param end_node: The ending node for path detection.
        :type end_node: Optional[int]
        :param max_path_length: Cutoff for path length during detection.
        :type max_path_length: Optional[int]

        :return: A string of concatenated feature values.
        :rtype: str
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
        """Finalize the hash using the features extracted and return the hash
        as a binary string.

        :param hash_object: The hash object.
        :type hash_object: Any
        :param features: Concatenated string of graph features.
        :type features: str

        :return: The final binary string of the hash, truncated or extended to `numBits`.
        :rtype: str
        """
        hash_object.update(features.encode())
        full_hash_binary = bin(int(hash_object.hexdigest(), 16))[2:]
        if len(full_hash_binary) < self.numBits:
            full_hash_binary += self.iterative_deepening(
                hash_object, self.numBits - len(full_hash_binary)
            )
        return full_hash_binary[: self.numBits]

    def iterative_deepening(self, hash_object: Any, remaining_bits: int) -> str:
        """Extend hash length using iterative hashing until the desired bit
        length is achieved.

        :param hash_object: The hash object for iterative deepening.
        :type hash_object: hashlib._Hash
        :param remaining_bits: Number of bits needed to reach `numBits`.
        :type remaining_bits: int

        :return: Additional binary data to achieve the desired hash length.
        :rtype: str
        """
        additional_data = ""
        while (
            len(additional_data) * 4 < remaining_bits
        ):  # Each hex digit represents 4 bits
            hash_object.update(additional_data.encode())
            additional_data += hash_object.hexdigest()
        return bin(int(additional_data, 16))[2:][:remaining_bits]
