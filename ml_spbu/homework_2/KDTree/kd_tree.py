"""Module implementing the KD-Tree data structure for efficient nearest neighbor search.

This module defines the KDTree class, which is used for organizing points in
k-dimensional space. KD-Trees are highly effective for nearest neighbor searches
and range queries, providing logarithmic time complexity on average for such operations.
The KDTree class utilizes the KDNode and KDMaxHeap classes from sibling modules
to build and query the tree structure.
"""

from ml_spbu.homework_2.annotations import CallableMetric, Point
from ml_spbu.homework_2.KDTree.kd_max_heap import KDMaxHeap
from ml_spbu.homework_2.KDTree.kd_node import KDNode


class KDTree:
    """KD-Tree data structure for efficient nearest neighbor search.

    This class constructs and queries a KD-Tree, a space-partitioning data
    structure for organizing points in a k-dimensional space. It is used for
    efficient nearest neighbor searches.

    Attributes:
        _root (KDNode): The root node of the KD-Tree.
        _metric (CallableMetric): The distance metric function used for distance calculations.
    """

    def __init__(self, X: list[Point], leaf_size: int, metric: CallableMetric):
        """Initialize KDTree by building the tree from the given data.

        Constructs a KD-Tree from the input data points `X`. The tree is built
        recursively, splitting nodes based on the median value along the axis
        with the largest spread, until the number of points in a node is less
        than or equal to `leaf_size`.

        Args:
            X (list[Point]): List of data points to build the KD-Tree from.
            leaf_size (int): The maximum number of points a leaf node can contain.
            metric (CallableMetric): The distance metric function to be used for
                nearest neighbor searches.
        """

        self._root: KDNode = KDNode(X, leaf_size)
        self._metric: CallableMetric = metric

    def knn_helper(self, x, k, current_node, k_nearest: KDMaxHeap):
        """Recursively traverse the KD-Tree to find k-nearest neighbors.

        This is a helper function for `_k_nearest_neighbors` and `query` methods.
        It recursively explores the KD-Tree to find the k-nearest neighbors of
        a given query point `x`. It uses a KDMaxHeap to maintain the k-nearest
        neighbors found so far.

        Args:
            x: (Point): The query point.
            k (int): The number of nearest neighbors to find.
            current_node (KDNode): The current node being explored in the KD-Tree.
            k_nearest (KDMaxHeap): The max heap to store the k-nearest neighbors found so far.
        """

        axis = current_node.axis
        axis_median = current_node.axis_median

        if current_node.is_leaf:
            for neighbor in current_node.X:
                k_nearest.add(neighbor)
            return

        if x[axis] < axis_median:
            first_subtree = current_node.left
            second_subtree = current_node.right
        else:
            first_subtree = current_node.right
            second_subtree = current_node.left

        if first_subtree:
            self.knn_helper(x, k, first_subtree, k_nearest)

        explore_other_subtree = (len(k_nearest.heap) < k) or (abs(x[axis] - axis_median) < k_nearest.heap[0][1])

        if second_subtree and explore_other_subtree:
            self.knn_helper(x, k, second_subtree, k_nearest)

    def _k_nearest_neighbors(self, x: Point, k: int, current_node: KDNode) -> list[Point]:
        """Find the k-nearest neighbors for a single query point using the KD-Tree.

        This method initiates the nearest neighbor search starting from the root
        of the KD-Tree. It utilizes the `knn_helper` method to recursively
        traverse the tree and find the k-nearest neighbors.

        Args:
            x (Point): The query point for which to find nearest neighbors.
            k (int): The number of nearest neighbors to retrieve.
            current_node (KDNode): The root node of the KD-Tree to start the search from.

        Returns:
            list[Point]: A list of the k-nearest neighbor points to the query point `x`.
                The neighbors are not guaranteed to be sorted by distance.
        """

        k_nearest = KDMaxHeap(x, k, self._metric)
        self.knn_helper(x, k, current_node, k_nearest)
        return [item[0] for item in k_nearest.heap]

    def query(self, X: list[Point], k: int) -> list[list[Point]]:
        """Find the k-nearest neighbors for each point in the query set X.

        For each query point in `X`, this method finds its k-nearest neighbors
        in the KD-Tree built from the training data. It returns a list of lists,
        where each inner list contains the k-nearest neighbors for the
        corresponding query point.

        Args:
            X (list[Point]): List of query points.
            k (int): The number of nearest neighbors to find for each query point.

        Returns:
            list[list[Point]]: A list of lists, where each inner list contains the
                k-nearest neighbors for the corresponding query point in `X`.
        """

        output = []

        for x in X:
            output.append(self._k_nearest_neighbors(x, k, self._root))

        return output
