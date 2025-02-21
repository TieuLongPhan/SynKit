import networkx as nx
import hashlib
import numpy as np


class GraphFP:
    def __init__(
        self, graph: nx.Graph, nBits: int = 1024, hash_alg: str = "sha256"
    ) -> None:
        """
        Initialize the GraphFP class to create binary fingerprints based on various graph
        characteristics.

        Parameters:
        - graph (nx.Graph): Graph on which to perform analysis.
        - nBits (int): Size of the binary fingerprint in bits.
        - hash_alg (str): Cryptographic hash function used for hashing.
        """
        self.graph = graph
        self.nBits = nBits
        self.hash_alg = hash_alg
        self.hash_function = getattr(hashlib, self.hash_alg)

    def fingerprint(self, method: str) -> str:
        """
        Generate a binary string fingerprint of the graph using the specified method.

        Parameters:
        - method (str): The method to use for fingerprinting
        ('spectrum', 'adjacency', 'degree', 'motif')

        Returns:
        - str: A binary string of length `nBits` that represents the fingerprint of
        the graph.
        """
        if method == "spectrum":
            fp = self._spectrum_fp()
        elif method == "adjacency":
            fp = self._adjacency_fp()
        elif method == "degree":
            fp = self._degree_sequence_fp()
        elif method == "motif":
            fp = self._motif_count_fp()
        else:
            raise ValueError("Unsupported fingerprinting method.")

        # If the fingerprint is shorter than nBits, use iterative deepening
        if len(fp) < self.nBits:
            fp += self.iterative_deepening(self.nBits - len(fp))

        return fp[: self.nBits]

    def _spectrum_fp(self) -> str:
        # Graph spectrum (eigenvalues of the adjacency matrix)
        eigenvalues = np.linalg.eigvals(nx.adjacency_matrix(self.graph).todense())
        sorted_eigenvalues = np.sort(eigenvalues)[: self.nBits]
        eigen_str = "".join(
            bin(int(abs(eig)))[2:].zfill(8) for eig in sorted_eigenvalues
        )
        return eigen_str[: self.nBits]

    def _adjacency_fp(self) -> str:
        # Adjacency matrix flattened
        adj_matrix = nx.adjacency_matrix(self.graph).todense().flatten()
        adj_str = "".join(str(int(x)) for x in adj_matrix)
        return adj_str[: self.nBits]

    def _degree_sequence_fp(self) -> str:
        # Degree sequence
        degrees = sorted([d for n, d in self.graph.degree()], reverse=True)
        degree_str = "".join(bin(d)[2:].zfill(8) for d in degrees)
        return degree_str[: self.nBits]

    def _motif_count_fp(self) -> str:
        # Motif counts (e.g., number of triangles)
        triangles = sum(nx.triangles(self.graph).values()) // 3
        triangle_str = bin(triangles)[2:].zfill(self.nBits)
        return triangle_str[: self.nBits]

    def iterative_deepening(self, remaining_bits: int) -> str:
        """
        Extend the hash length using iterative hashing until the desired bit length is
        achieved.

        Parameters:
        - remaining_bits (int): Number of bits needed to complete the fingerprint
        to `nBits`.

        Returns:
        - str: Additional binary data to achieve the desired hash length.
        """
        additional_data = ""
        hash_obj = self.hash_function()
        while len(additional_data) * 4 < remaining_bits:
            hash_obj.update(additional_data.encode())
            additional_data += hash_obj.hexdigest()
        return bin(int(additional_data, 16))[2:][:remaining_bits]
