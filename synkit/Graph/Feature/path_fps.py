import networkx as nx
import hashlib
from typing import Any


class PathFPs:
    def __init__(
        self,
        graph: nx.Graph,
        max_length: int = 10,
        nBits: int = 1024,
        hash_alg: str = "sha256",
    ) -> None:
        """Initialize the PathFPs class to create a binary fingerprint based on
        paths in a graph.

        :param graph: Graph on which to perform analysis.
        :type graph: nx.Graph
        :param max_length: Limit on path lengths considered in the fingerprint.
        :type max_length: int
        :param nBits: Size of the binary fingerprint in bits.
        :type nBits: int
        :param hash_alg: Cryptographic hash function used for path hashing.
        :type hash_alg: str
        """
        self.graph = graph
        self.max_length = max_length
        self.nBits = nBits
        self.hash_alg = hash_alg
        self.hash_function = getattr(hashlib, self.hash_alg)

    def generate_fingerprint(self) -> str:
        """Generate a binary string fingerprint of the graph by hashing paths
        up to a certain length and combining them.

        :return: A binary string of length `nBits` that represents the fingerprint of the graph.
        :rtype: str
        """
        fingerprint = ""
        for node in self.graph.nodes():
            for target in self.graph.nodes():
                if node != target:
                    for path in nx.all_simple_paths(
                        self.graph, source=node, target=target, cutoff=self.max_length
                    ):
                        path_str = "-".join(map(str, path))
                        hash_obj = self.hash_function(path_str.encode())
                        path_hash = bin(int(hash_obj.hexdigest(), 16))[2:].zfill(
                            hash_obj.digest_size * 8
                        )
                        if len(fingerprint) + len(path_hash) > self.nBits:
                            needed_bits = self.nBits - len(fingerprint)
                            path_hash = path_hash[:needed_bits]
                        fingerprint += path_hash
                        if len(fingerprint) == self.nBits:
                            return fingerprint

        if len(fingerprint) < self.nBits:
            fingerprint += self.iterative_deepening(
                hash_obj, self.nBits - len(fingerprint)
            )
        return fingerprint

    def iterative_deepening(self, hash_object: Any, remaining_bits: int) -> str:
        """Extend the hash length using iterative hashing until the desired bit
        length is achieved.

        :param hash_object: The hash object used for iterative deepening.
        :type hash_object: hashlib._Hash
        :param remaining_bits: Number of bits needed to complete the fingerprint
                               to `nBits`.
        :type remaining_bits: int

        :return: Additional binary data to achieve the desired hash length.
        :rtype: str
        """
        additional_data = ""
        while len(additional_data) * 4 < remaining_bits:
            hash_object.update(additional_data.encode())
            additional_data += hash_object.hexdigest()
        return bin(int(additional_data, 16))[2:][:remaining_bits]
