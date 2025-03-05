"""Module implementing a Max Heap for KD-Tree Nearest Neighbor Search.

This module defines the KDMaxHeap class, which is used to efficiently keep
track of the k-nearest neighbors during the traversal of a KD-Tree. It
prioritizes points with larger distances, allowing for quick access to the
maximum distance among the current k-nearest neighbors.
"""

from ml_spbu.homework_2.annotations import CallableMetric, Point


class KDMaxHeap:
    """Max Heap data structure for storing and managing k-nearest neighbors.

    This class implements a max heap to store points and their distances to a
    fixed point. It is designed to maintain a heap of size at most 'k', where
    the top element is the point with the largest distance. This is useful in
    k-NN search algorithms to efficiently keep track of the k nearest points
    found so far.

    Attributes:
        heap (list[tuple[Point, float]]): A list representing the max heap. Each element
            is a tuple containing a Point and its distance to fixed point.
        _metric (CallableMetric): The distance metric function used to calculate distances.
        _fixed_x (Point): The reference point for distance calculations.
        _capacity (int): The maximum capacity of the heap (k-value for k-NN).
    """

    def __init__(self, x: Point, k: int, metric: CallableMetric):
        """Initialize KDMaxHeap with a reference point, capacity, and metric.

        Args:
            x (Point): The fixed point to which distances are calculated.
            k (int): The maximum number of elements (neighbors) the heap can hold.
            metric (CallableMetric): The distance metric function to be used.
        """

        self.heap: list[tuple[Point, float]] = []
        self._metric = metric
        self._fixed_x = x
        self._capacity = k

    def heapify(self, i: int):
        """Heapify a subtree rooted at index i.

        This method maintains the max heap property starting from index i down to the leaves.
        It is used after insertion and deletion to restore the heap structure.
        Works only if all subtrees are maxheaps.

        Args:
            i (int): The index of the root of the subtree to heapify.
        """

        heap = self.heap
        n = len(heap)
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and heap[left][1] > heap[largest][1]:
            largest = left

        if right < n and heap[right][1] > heap[largest][1]:
            largest = right

        if largest != i:
            heap[i], heap[largest] = heap[largest], heap[i]
            self.heapify(largest)

    def _delete_max(self):
        """Delete the maximum element (root) from the max heap.

        This is a helper method to remove the element with the largest distance
        from the heap, which is always at the root (index 0) in a max heap.
        It is used to maintain the heap size at most 'k' by removing the
        farthest neighbor when a new, closer neighbor is found.
        """

        heap = self.heap

        if heap:
            heap[0] = heap[-1]
            heap.pop()

            self.heapify(0)

    def add(self, x: Point):
        """Add a point to the max heap if it's among the k-nearest.

        This method adds a point 'x' to the heap if it is closer to the fixed point
        than the current farthest neighbor in the heap (if the heap is full) or if
        the heap is not yet full. If the heap is full and 'x' is closer than the
        farthest neighbor, the farthest neighbor is removed, and 'x' is added.

        Args:
            x (Point): The point to be added to the heap.
        """

        heap = self.heap
        fixed_x = self._fixed_x
        metric = self._metric

        dist_x_fixed_x = metric(x, fixed_x)

        if not heap:
            heap.append((x, dist_x_fixed_x))
            return None

        if (dist_x_fixed_x > heap[0][1]) and (len(heap) == self._capacity):
            return None

        if len(heap) == self._capacity:
            self._delete_max()

        heap.append((x, dist_x_fixed_x))
        i = len(heap) - 1
        parent = (i - 1) // 2

        while (i > 0) and (heap[parent][1] < heap[i][1]):
            heap[i], heap[parent] = heap[parent], heap[i]

            i = parent
            parent = (i - 1) // 2
