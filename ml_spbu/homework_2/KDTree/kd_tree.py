from ml_spbu.homework_2.annotations import CallableMetric, Point
from ml_spbu.homework_2.KDTree.kd_max_heap import KDMaxHeap
from ml_spbu.homework_2.KDTree.kd_node import KDNode


class KDTree:
    def __init__(self, X: list[Point], leaf_size: int, metric: CallableMetric):
        self._root: KDNode = KDNode(X, leaf_size)
        self._metric: CallableMetric = metric

    def knn_helper(self, x, k, current_node, k_nearest: KDMaxHeap):
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
        k_nearest = KDMaxHeap(x, k, self._metric)
        self.knn_helper(x, k, current_node, k_nearest)
        return [item[0] for item in k_nearest.heap]

    def query(self, X: list[Point], k: int) -> list[list[Point]]:
        output = []

        for x in X:
            output.append(self._k_nearest_neighbors(x, k, self._root))

        return output
