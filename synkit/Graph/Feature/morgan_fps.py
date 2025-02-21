import networkx as nx
import hashlib
from typing import Any


class MorganFPs:
    def __init__(
        self,
        graph: nx.Graph,
        radius: int = 3,
        nBits: int = 1024,
        hash_alg: str = "sha256",
    ):
        """
        Initialize the MorganFPs class to generate fingerprints based on the Morgan
        algorithm, approximating Extended Connectivity Fingerprints (ECFPs).

        Parameters:
        - graph (nx.Graph): The graph to analyze.
        - radius (int): The radius to consider for node neighborhood analysis.
        - nBits (int): Total number of bits in the final fingerprint output.
        - hash_alg (str): Hash algorithm to use for generating hashes of node
        neighborhoods.
        """
        self.graph = graph
        self.radius = radius
        self.nBits = nBits
        self.hash_alg = hash_alg
        self.hash_function = getattr(hashlib, self.hash_alg)

    def generate_fingerprint(self) -> str:
        """
        Generate a binary string fingerprint of the graph based on the local environments
        of nodes. Ensures the output is exactly `nBits` in length using iterative
        deepening if necessary.

        Returns:
        - str: A binary string of length `nBits` representing the fingerprint of the
        graph.
        """
        fingerprint = ""
        for node in self.graph.nodes():
            neighborhood = nx.single_source_shortest_path_length(
                self.graph, node, cutoff=self.radius
            )
            neighborhood_str = "-".join(
                [
                    f"{nbr}-{dist}"
                    for nbr, dist in sorted(neighborhood.items())
                    if nbr != node
                ]
            )
            hash_obj = self.hash_function(neighborhood_str.encode())
            node_hash = bin(int(hash_obj.hexdigest(), 16))[2:].zfill(
                hash_obj.digest_size * 8
            )
            if len(fingerprint) + len(node_hash) > self.nBits:
                needed_bits = self.nBits - len(fingerprint)
                node_hash = node_hash[:needed_bits]
            fingerprint += node_hash
            if len(fingerprint) == self.nBits:
                return fingerprint

        if len(fingerprint) < self.nBits:
            fingerprint += self.iterative_deepening(
                hash_obj, self.nBits - len(fingerprint)
            )
        return fingerprint

    def iterative_deepening(self, hash_object: Any, remaining_bits: int) -> str:
        """
        Extend the hash length using iterative hashing until the desired bit length is
        achieved.

        Parameters:
        - hash_object (hashlib._Hash): The hash object used for iterative deepening.
        - remaining_bits (int): Number of bits needed to complete the fingerprint to
        `nBits`.

        Returns:
        - str: Additional binary data to achieve the desired hash length.
        """
        additional_data = ""
        while len(additional_data) * 4 < remaining_bits:
            hash_object.update(additional_data.encode())
            additional_data += hash_object.hexdigest()
        return bin(int(additional_data, 16))[2:][:remaining_bits]
