import pytest
from hypothesis import given
import hypothesis.strategies as st

from ml_spbu.homework_2.KDTree.kd_max_heap import KDMaxHeap


def euclidean_distance(p1, p2) -> float:
    return sum([(x - y) ** 2 for x, y in zip(p1, p2)]) ** 0.5


class TestKDMaxHeap:
    @pytest.fixture
    def fixed_point(self):
        return 0, 0

    @pytest.fixture
    def metric_func(self):
        return euclidean_distance

    def test_init(self, fixed_point, metric_func):
        heap = KDMaxHeap(fixed_point, 3, metric_func)
        assert heap.heap == []
        assert heap._metric == metric_func
        assert heap._fixed_x == fixed_point
        assert heap._capacity == 3

    def test_add_to_empty_heap(self, fixed_point, metric_func):
        heap = KDMaxHeap(fixed_point, 3, metric_func)
        point1 = (1, 1)
        heap.add(point1)
        assert len(heap.heap) == 1
        assert heap.heap[0] == (point1, metric_func(point1, fixed_point))

    def test_add_multiple_points(self, fixed_point, metric_func):
        heap = KDMaxHeap(fixed_point, 3, metric_func)
        points = [(1, 0), (0, 1), (1, 1), (0.5, 0.5)]
        for point in points:
            heap.add(point)

        assert len(heap.heap) == 3
        heap_distances = [dist for _, dist in heap.heap]
        assert all(heap_distances[i] >= heap_distances[2 * i + 1] for i in range(len(heap_distances) // 2))
        assert all(
            heap_distances[i] >= heap_distances[2 * i + 2]
            for i in range(len(heap_distances) // 2)
            if 2 * i + 2 < len(heap_distances)
        )

    def test_add_point_outside_max_distance_full(self, fixed_point, metric_func):
        heap = KDMaxHeap(fixed_point, 2, metric_func)
        point1 = (1, 0)
        point2 = (0, 1)
        heap.add(point1)
        heap.add(point2)
        assert len(heap.heap) == 2
        root_before = heap.heap[0]

        point3 = (2, 2)
        heap.add(point3)
        assert len(heap.heap) == 2
        assert heap.heap[0] == root_before

    def test_add_point_inside_max_distance_heap_full(self, fixed_point, metric_func):
        heap = KDMaxHeap(fixed_point, 2, metric_func)
        point1 = (1, 0)
        point2 = (0, 4)
        heap.add(point1)
        heap.add(point2)
        assert len(heap.heap) == 2

        point3 = (0.1, 0.1)
        point4 = (0, 0.5)
        heap.add(point3)
        heap.add(point4)
        assert len(heap.heap) == 2
        assert heap.heap[0] == (point4, metric_func(point4, fixed_point))

    def test_heapify_basic(self, fixed_point, metric_func):
        heap_instance = KDMaxHeap(fixed_point, 5, metric_func)
        heap_instance.heap = [((1, 1), 1.41), ((3, 3), 4.24), ((0.5, 0.5), 0.7), ((2, 2), 2.82), ((0, 0), 0.0)]
        heap_instance.heapify(0)
        heap = heap_instance.heap
        heap_distances = [dist for _, dist in heap]
        assert all(
            heap_distances[i] >= heap_distances[2 * i + 1]
            for i in range(len(heap_distances) // 2)
            if 2 * i + 1 < len(heap_distances)
        )
        assert all(
            heap_distances[i] >= heap_distances[2 * i + 2]
            for i in range(len(heap_distances) // 2)
            if 2 * i + 2 < len(heap_distances)
        )

    def test_heapify_empty_heap(self, fixed_point, metric_func):
        heap = KDMaxHeap(fixed_point, 5, metric_func)
        heap.heapify(0)
        assert heap.heap == []

    def test_delete_max_basic(self, fixed_point, metric_func):
        heap = KDMaxHeap(fixed_point, 5, metric_func)
        heap.heap = [((3, 3), 4.24), ((2, 2), 2.82), ((1, 1), 1.41), ((0.5, 0.5), 0.7), ((0, 0), 0.0)]
        root_before = heap.heap[0]
        heap._delete_max()
        assert root_before not in heap.heap

        heap_distances = [dist for _, dist in heap.heap]
        assert all(
            heap_distances[i] >= heap_distances[2 * i + 1]
            for i in range(len(heap_distances) // 2)
            if 2 * i + 1 < len(heap_distances)
        )
        assert all(
            heap_distances[i] >= heap_distances[2 * i + 2]
            for i in range(len(heap_distances) // 2)
            if 2 * i + 2 < len(heap_distances)
        )

    def test_delete_max_single_element(self, fixed_point, metric_func):
        heap = KDMaxHeap(fixed_point, 5, metric_func)
        heap.heap = [((1, 1), 1.41)]
        heap._delete_max()
        assert heap.heap == []

    def test_delete_max_empty_heap(self, fixed_point, metric_func):
        heap = KDMaxHeap(fixed_point, 5, metric_func)
        heap._delete_max()
        assert heap.heap == []

    @given(
        data=st.data(), k=st.integers(min_value=1, max_value=40), points_count=st.integers(min_value=0, max_value=1000)
    )
    def test_hypothesis_add_multiple_random_points(self, data, k, points_count):
        heap = KDMaxHeap((0, 0), k, euclidean_distance)
        points_strategy = st.lists(
            st.tuples(
                st.floats(min_value=-1000, max_value=1000, allow_nan=False),
                st.floats(min_value=-1000, max_value=1000, allow_nan=False),
            ),
            min_size=points_count,
            max_size=points_count,
        )
        generated_points = data.draw(points_strategy)

        for point in generated_points:
            heap.add(point)

        assert len(heap.heap) <= k

        heap_distances = [dist for _, dist in heap.heap]
        if heap_distances:
            assert all(
                heap_distances[i] >= heap_distances[2 * i + 1]
                for i in range(len(heap_distances) // 2)
                if 2 * i + 1 < len(heap_distances)
            )
            assert all(
                heap_distances[i] >= heap_distances[2 * i + 2]
                for i in range(len(heap_distances) // 2)
                if 2 * i + 2 < len(heap_distances)
            )
