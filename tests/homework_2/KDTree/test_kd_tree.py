import random

import pytest
import hypothesis.strategies as st
from hypothesis import given

from ml_spbu.homework_2.KDTree.kd_node import KDNode
from ml_spbu.homework_2.KDTree.kd_tree import KDTree
from ml_spbu.homework_2.annotations import Point


def filter_floats(x):
    return abs(x) > 0.2


def euclidean_distance(point1: Point, point2: Point) -> float:
    return sum([(p1 - p2) ** 2 for p1, p2 in zip(point1, point2)]) ** 0.5


def manhattan_distance(point1: Point, point2: Point) -> float:
    return sum([abs(p1 - p2) for p1, p2 in zip(point1, point2)])


class TestKDTree:
    def test_kd_tree_init_valid_x(self):
        points = [(1, 2), (3, 4), (5, 6)]
        tree = KDTree(points, 1, euclidean_distance)
        assert tree._root is not None
        assert tree._root.left is not None
        assert tree._root.right is not None
        assert isinstance(tree._root, KDNode)

    def test_kd_tree_init_invalid_leaf_size(self):
        points = [(1, 2), (3, 4), (5, 6)]
        with pytest.raises(AttributeError):
            KDTree(points, 0, euclidean_distance)
        with pytest.raises(AttributeError):
            KDTree(points, -1, euclidean_distance)

    @given(
        X=st.lists(
            st.tuples(
                st.floats(min_value=-100, max_value=100),
                st.floats(min_value=-100, max_value=100),
            ),
            min_size=5,
        ),
        leaf_size=st.integers(min_value=1, max_value=10),
        query_points=st.lists(
            st.tuples(
                st.floats(min_value=-100, max_value=100),
                st.floats(min_value=-100, max_value=100),
            ),
            min_size=1,
        ),
        k=st.integers(min_value=1, max_value=5),
    )
    def test_kd_tree_k_neighbors_count(self, X, leaf_size, query_points, k):
        tree = KDTree(X, leaf_size, euclidean_distance)
        results = tree.query(query_points, k)
        assert len(results) == len(query_points)
        for neighbors in results:
            assert len(neighbors) == k

    @given(
        X=st.lists(
            st.tuples(
                st.floats(min_value=-100, max_value=100),
                st.floats(min_value=-100, max_value=100),
            ),
            min_size=5,
        ),
        leaf_size=st.integers(min_value=1, max_value=5),
        query_point=st.tuples(
            st.floats(min_value=-100, max_value=100),
            st.floats(min_value=-100, max_value=100),
        ),
        k=st.integers(min_value=1, max_value=5),
    )
    def test_kd_tree_nearest_neighbors_distance(self, X, leaf_size, query_point, k):
        tree = KDTree(X, leaf_size, metric=euclidean_distance)
        neighbors = tree.query([query_point], k)[0]
        neighbor_distances = sorted([euclidean_distance(query_point, neighbor) for neighbor in neighbors])

        distances_to_all_points = sorted([(euclidean_distance(query_point, p), p) for p in X])
        expected_neighbors = [p for dist, p in distances_to_all_points[:k]]
        expected_neighbor_distances = sorted(
            [euclidean_distance(query_point, neighbor) for neighbor in expected_neighbors]
        )

        assert len(neighbors) == k

        tolerance = 1e-6
        for i in range(len(neighbors)):
            assert abs(neighbor_distances[i] - expected_neighbor_distances[i]) <= tolerance

    @given(
        X=st.lists(
            st.tuples(st.floats(min_value=-100, max_value=100), st.floats(min_value=-100, max_value=100)),
            min_size=1,
        ),
        leaf_size=st.integers(min_value=1, max_value=10),
        query_point=st.tuples(st.floats(min_value=-100, max_value=100), st.floats(min_value=-100, max_value=100)),
        k=st.integers(min_value=1, max_value=10),
    )
    def test_kd_tree_query_property_neighbors_are_from_dataset(self, X, leaf_size, query_point, k):
        tree = KDTree(X, leaf_size, metric=euclidean_distance)
        neighbors = tree.query([query_point], k)[0]

        for neighbor in neighbors:
            assert neighbor in X
