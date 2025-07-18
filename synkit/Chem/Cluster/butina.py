from __future__ import annotations
from typing import List, Optional

import numpy as np
from rdkit.DataStructs import cDataStructs, CreateFromBitString, BulkTanimotoSimilarity
from rdkit.ML.Cluster import Butina
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class ButinaCluster:
    """Cluster chemical fingerprint vectors using the Butina algorithm from
    RDKit, with integrated t-SNE visualization of clusters.

    Key features
    ------------
    * **Butina clustering** – fast hierarchical clustering with a similarity cutoff.
    * **t-SNE visualization** – 2D embedding of fingerprints, highlighting top‑k clusters.
    * **NumPy support** – accepts 2D arrays of 0/1 fingerprint data.
    * **Configurable** – user‑defined cutoff, perplexity, and top‑k highlight.

    Quick start
    -----------
    >>> from synkit.Chem.Fingerprint.fingerprint_clusterer import ButinaCluster
    >>> clusters = ButinaCluster.cluster(arr, cutoff=0.3)
    >>> ButinaCluster.visualize(arr, clusters, k=5)
    """

    @staticmethod
    def cluster(arr: np.ndarray, cutoff: float = 0.2) -> List[List[int]]:
        """Perform Butina clustering on fingerprint bit-vectors.

        :param arr: 2D array of shape (n_samples, n_bits) with 0/1
            dtype.
        :type arr: np.ndarray
        :param cutoff: Distance cutoff (1 – similarity) to form
            clusters. Defaults to 0.2.
        :type cutoff: float
        :returns: List of clusters, each a list of sample indices.
        :rtype: list of list of int
        """
        # Convert rows to RDKit ExplicitBitVect
        fps: List[cDataStructs.ExplicitBitVect] = []
        for row in arr:
            bitstr = "".join(str(int(b)) for b in row.tolist())
            fps.append(CreateFromBitString(bitstr))

        n = len(fps)
        # Build flattened upper‐triangular distance list
        distances: List[float] = []
        for i in range(n):
            # fmt: off
            sims = BulkTanimotoSimilarity(fps[i], fps[i + 1:])
            # fmt: on
            distances.extend((1.0 - np.array(sims, dtype=float)).tolist())

        # Cluster: ClusterData(distanceList, nPts, cutoff, isDistData)
        clusters = Butina.ClusterData(distances, n, cutoff, True)
        return clusters

    @staticmethod
    def visualize(
        arr: np.ndarray,
        clusters: List[List[int]],
        k: Optional[int] = None,
        perplexity: float = 30.0,
        random_state: int = 42,
    ) -> None:
        """Visualize clusters in 2D via t-SNE embedding.

        :param arr: 2D array of shape (n_samples, n_features) with fingerprint data.
        :type arr: np.ndarray
        :param clusters: Clusters as returned by `cluster()`.
        :type clusters: list of list of int
        :param k: If provided, highlight only the top‑k largest clusters; others shown as 'Other'.
        :type k: int or None
        :param perplexity: t-SNE perplexity parameter. Defaults to 30.0.
        :type perplexity: float
        :param random_state: Random seed for reproducibility. Defaults to 42.
        :type random_state: int
        :returns: None
        :rtype: NoneType

        :example:
        >>> clusters = ButinaCluster.cluster(arr, cutoff=0.3)
        >>> ButinaCluster.visualize(arr, clusters, k=5)
        """
        n = arr.shape[0]
        # assign labels: cluster idx or -1 for 'Other'
        labels = np.full(n, -1, dtype=int)
        # sort clusters by size
        sorted_idx = sorted(
            range(len(clusters)), key=lambda i: len(clusters[i]), reverse=True
        )
        top = set(sorted_idx[:k]) if k is not None else set(sorted_idx)
        for idx, cluster in enumerate(clusters):
            for i in cluster:
                labels[i] = idx if idx in top else -1

        # compute t-SNE embedding
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        emb = tsne.fit_transform(arr)

        # plot
        plt.figure(figsize=(8, 6))
        unique = sorted(set(labels))
        for lab in unique:
            mask = labels == lab
            if lab == -1:
                plt.scatter(
                    emb[mask, 0], emb[mask, 1], color="gray", alpha=0.3, label="Other"
                )
            else:
                plt.scatter(
                    emb[mask, 0], emb[mask, 1], alpha=0.7, label=f"Cluster {lab}"
                )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.title("t-SNE visualization of Butina clusters")
        plt.xlabel("t-SNE dim 1")
        plt.ylabel("t-SNE dim 2")
        plt.tight_layout()
        plt.show()

    def __str__(self) -> str:
        """Short description of the clusterer.

        :returns: Class name.
        :rtype: str
        """
        return "<ButinaCluster>"

    def help(self) -> None:
        """Print usage summary for clustering and visualization.

        :returns: None
        :rtype: NoneType
        """
        print("ButinaCluster.cluster(arr, cutoff=0.2)")
        print("ButinaCluster.visualize(arr, clusters, k=None, perplexity=30.0)")
