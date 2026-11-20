import math

import pytest

import occlusion

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

R1 = (0, 0, 10, 10)
R2 = (2, 2, 4, 4)
R3 = (20, 20, 5, 5)


def _check_occl_inside():
    return occlusion.Occluder().cull([R1, R2]) == [R1]


def _check_occl_outside():
    return occlusion.Occluder().cull([R1, R3]) == [R1, R3]


def _check_occl_partial():
    return occlusion.Occluder().cull([R1, (8, 8, 5, 5)]) == [R1, (8, 8, 5, 5)]


def _check_occl_empty():
    return occlusion.Occluder().cull([]) == []


def _check_occl_single():
    return occlusion.Occluder().cull([R1]) == [R1]


def _check_occl_order():
    a = occlusion.Occluder().cull([R1, R2, R3])
    b = occlusion.Occluder().cull([R3, R2, R1])
    return sorted(a) == sorted(b)


def _check_occl_stats():
    o = occlusion.Occluder()
    o.cull([R1, R2])
    return o.stats()["culled"] == 1


def _check_occl_bad():
    return _raises(lambda: occlusion.Occluder().cull([(5, 5, -1, 1)]))


def _check_occl_nested():
    return occlusion.Occluder().cull([R1, (1, 1, 2, 2), (3, 3, 2, 2)]) == [R1]


def _check_occl_touching():
    return occlusion.Occluder().cull([R1, (10, 0, 5, 5)]) == [R1, (10, 0, 5, 5)]


def _check_occl_degenerate():
    return _raises(lambda: occlusion.Occluder().cull([(0, 0, 0, 0)]))


def _check_occl_count():
    return len(occlusion.Occluder().cull([R1, R2, R3])) == 2


def _check_occl_det():
    return occlusion.Occluder().cull([R1, R2]) == occlusion.Occluder().cull([R1, R2])


def _check_occl_type():
    return isinstance(occlusion.Occluder().cull([R1]), list)


def _check_occl_no_mutation():
    rs = [R1, R2]
    occlusion.Occluder().cull(rs)
    return rs == [R1, R2]

CASES = [("occl_inside", "_check_occl_inside"), ("occl_outside", "_check_occl_outside"), ("occl_partial", "_check_occl_partial"), ("occl_empty", "_check_occl_empty"), ("occl_single", "_check_occl_single"), ("occl_order", "_check_occl_order"), ("occl_stats", "_check_occl_stats"), ("occl_bad", "_check_occl_bad"), ("occl_nested", "_check_occl_nested"), ("occl_touching", "_check_occl_touching"), ("occl_degenerate", "_check_occl_degenerate"), ("occl_count", "_check_occl_count"), ("occl_det", "_check_occl_det"), ("occl_type", "_check_occl_type"), ("occl_no_mutation", "_check_occl_no_mutation")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
