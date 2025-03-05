from ml_spbu.homework_2.annotations import CallableMetric, Point


class KDMaxHeap:
    def __init__(self, x: Point, k: int, metric: CallableMetric):
        self.heap: list[tuple[Point, float]] = []
        self._metric = metric
        self._fixed_x = x
        self._capacity = k

    def heapify(self, i: int):
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
        heap = self.heap

        if heap:
            heap[0] = heap[-1]
            heap.pop()

            self.heapify(0)

    def add(self, x: Point):
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
