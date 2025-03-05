import pytest
from hypothesis import given
import hypothesis.strategies as st

from ml_spbu.homework_2.KDTree.kd_node import KDNode
from ml_spbu.homework_2.annotations import Point


def points_strategy(dimensions):
    return st.lists(
        st.tuples(*[st.floats(allow_nan=False, allow_infinity=False) for _ in range(dimensions)]), min_size=1
    )


def collect_leaf_points(node: KDNode) -> list[Point]:
    if node.is_leaf:
        return node.X
    else:
        left_points = collect_leaf_points(node.left) if node.left else []
        right_points = collect_leaf_points(node.right) if node.right else []
        return left_points + right_points


class TestKDNode:
    def test_init_leaf_node(self):
        X = [(1, 2), (3, 4), (5, 6)]
        leaf_size = 5
        node = KDNode(X, leaf_size)
        assert node.is_leaf is True
        assert node.X == X
        assert node.axis is None
        assert node.axis_median is None
        assert node.left is None
        assert node.right is None

    def test_init_internal_node(self):
        X = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
        leaf_size = 2
        node = KDNode(X, leaf_size)
        assert node.is_leaf is False
        assert node.axis is not None
        assert node.axis_median is not None
        assert node.left is not None
        assert node.right is not None
        assert isinstance(node.left, KDNode)
        assert isinstance(node.right, KDNode)

    def test_choose_axis_2d(self):
        """Test _choose_axis method in 2D."""
        X = [(1, 2), (3, 1), (5, 4), (2, 3)]
        node = KDNode(X, 3)
        axis = node._choose_axis(X)
        assert axis == 0

        X_equal_spread = [(0, 0), (1, 1), (0, 1), (1, 0)]
        node_equal_spread = KDNode(X_equal_spread, 3)
        axis_equal_spread = node_equal_spread._choose_axis(X_equal_spread)
        assert axis_equal_spread == 0

    def test_choose_axis_3d(self):
        X = [(1, 2, 3), (4, 2, 1), (2, 5, 2), (3, 3, 6)]
        node = KDNode(X, 3)
        axis = node._choose_axis(X)
        assert axis == 2

    def test_split_2d(self):
        X = [(1, 2), (3, 4), (5, 6), (2, 3), (4, 5)]
        node = KDNode(X, 10)
        axis = 0
        median_val, left_X, right_X = node._split(X, axis)
        assert median_val == 3
        assert left_X == [(1, 2), (2, 3)]
        assert right_X == [(3, 4), (5, 6), (4, 5)]

        axis_y = 1
        median_val_y, left_X_y, right_X_y = node._split(X, axis_y)
        assert median_val_y == 4
        assert left_X_y == [(1, 2), (2, 3)]
        assert right_X_y == [(3, 4), (5, 6), (4, 5)]

    def test_split_empty_side_handling(self):
        X_almost_same_x = [(3, 2), (3, 4), (3, 6), (3, 3), (4, 5)]
        node = KDNode(X_almost_same_x, 10)
        axis = 0
        median_val, left_X, right_X = node._split(X_almost_same_x, axis)
        assert median_val == 3.5
        assert left_X == [(3, 2), (3, 4), (3, 6), (3, 3)]
        assert right_X == [(4, 5)]

        X_same_x = [(3, 2), (3, 4), (3, 6), (3, 3)]
        node_same_x = KDNode(X_same_x, 10)
        median_val_same, left_X_same, right_X_same = node_same_x._split(X_same_x, axis)
        assert median_val_same == 3
        assert left_X_same == [(3, 2), (3, 4)]
        assert right_X_same == [(3, 6), (3, 3)]

    def test_init_invalid_leaf_size(self):
        X = [(1, 2), (3, 4)]
        with pytest.raises(AttributeError, match="Leaf size must be strictly positive"):
            KDNode(X, 0)

        with pytest.raises(AttributeError, match="Leaf size must be strictly positive"):
            KDNode(X, -1)

    def test_init_empty_X_leaf(self):
        X: list[Point] = []
        leaf_size = 5
        node = KDNode(X, leaf_size)
        assert node.is_leaf is True
        assert node.X == X
        assert node.axis is None
        assert node.axis_median is None
        assert node.left is None
        assert node.right is None

    @given(leaf_size=st.integers(min_value=1, max_value=10), points_list=points_strategy(dimensions=5))
    def test_kd_node_all_points_in_leaves_hypothesis(self, leaf_size, points_list):
        node = KDNode(points_list, leaf_size)
        collected_points = collect_leaf_points(node)

        assert set(collected_points) == set(points_list)
