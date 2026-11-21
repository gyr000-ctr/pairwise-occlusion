"""题面边界与隐藏用例维度的补充测试。

不改动既有 tests/test_core.py；这里覆盖：可区分的错误类型、并发与
串行一致、极大坐标、网格重叠、空输入、顺序无关、stats 累计语义。
"""

import math
import random
import threading

import pytest

import occlusion
from occlusion.core import DegenerateRectError, NonFiniteCoordinateError


def _brute(rects):
    """两两包含参考实现（基于集合，顺序无关）。"""
    norm = [(x, y, x + w, y + h) for (x, y, w, h) in rects]
    kept = set()
    for i, b in enumerate(norm):
        culled = False
        for j, o in enumerate(norm):
            if i == j:
                continue
            if (o[0] <= b[0] and o[1] <= b[1]
                    and o[2] >= b[2] and o[3] >= b[3]):
                culled = True
                break
        if not culled:
            kept.add(b)
    return {r for r, b in zip(rects, norm) if b in kept}


def test_empty_returns_empty_list():
    out = occlusion.Occluder().cull([])
    assert out == []
    assert isinstance(out, list)


def test_distinct_error_types_are_distinguishable():
    # 退化矩形与非有限坐标抛出不同的、可区分的异常类型。
    with pytest.raises(DegenerateRectError):
        occlusion.Occluder().cull([(0, 0, 0, 5)])
    with pytest.raises(DegenerateRectError):
        occlusion.Occluder().cull([(0, 0, 5, -1)])
    with pytest.raises(NonFiniteCoordinateError):
        occlusion.Occluder().cull([(0, 0, float("nan"), 5)])
    with pytest.raises(NonFiniteCoordinateError):
        occlusion.Occluder().cull([(float("inf"), 0, 5, 5)])
    with pytest.raises(NonFiniteCoordinateError):
        occlusion.Occluder().cull([(0, 0, float("inf"), 5)])
    # 两者仍是 ValueError，且彼此不同、可区分。
    assert issubclass(DegenerateRectError, ValueError)
    assert issubclass(NonFiniteCoordinateError, ValueError)
    assert DegenerateRectError is not NonFiniteCoordinateError


def test_validation_runs_before_any_culling():
    # 含非法元素时整体报错，不返回部分结果。
    with pytest.raises(NonFiniteCoordinateError):
        occlusion.Occluder().cull([(0, 0, 10, 10), (1, 1, float("nan"), 1)])


def test_huge_coordinates_exact_containment():
    big = 10 ** 15
    outer = (big, big, 10 ** 12, 10 ** 12)
    inner = (big + 1, big + 1, 1, 1)
    assert occlusion.Occluder().cull([outer, inner]) == [outer]


def test_grid_partial_overlap_all_kept():
    # 步长 3、边长 5：相邻矩形部分重叠但互不包含。
    grid = [(x * 3, y * 3, 5, 5) for x in range(20) for y in range(20)]
    out = occlusion.Occluder().cull(grid)
    assert sorted(out) == sorted(grid)
    assert _brute(grid) == set(grid)


def test_touching_edges_not_contained():
    base = (0, 0, 10, 10)
    touch = (10, 0, 5, 5)  # 仅共边
    assert occlusion.Occluder().cull([base, touch]) == [base, touch]


def test_order_independence_matches_pairwise():
    rng = random.Random(2026)
    for _ in range(50):
        rects = [(rng.randint(0, 12), rng.randint(0, 12),
                  rng.randint(1, 8), rng.randint(1, 8))
                 for _ in range(rng.randint(1, 80))]
        ref = _brute(rects)
        out_a = occlusion.Occluder().cull(rects)
        shuffled = rects[:]
        rng.shuffle(shuffled)
        out_b = occlusion.Occluder().cull(shuffled)
        assert set(out_a) == ref
        assert set(out_b) == ref
        # 输出遵循各自的输入相对顺序。
        assert out_a == [r for r in rects if r in set(out_a)]


def test_duplicate_rectangles_all_culled():
    # 两两语义下，相同矩形互相完全包含，所有副本都被剔除。
    rects = [(0, 0, 5, 5), (0, 0, 5, 5), (0, 0, 5, 5)]
    assert occlusion.Occluder().cull(rects) == []
    # 但该几何仍能遮挡更小的矩形。
    assert occlusion.Occluder().cull(rects + [(1, 1, 1, 1)]) == []


def test_concurrent_same_as_serial():
    rng = random.Random(77)
    rects = [(rng.randint(0, 30), rng.randint(0, 30),
              rng.randint(1, 20), rng.randint(1, 20))
             for _ in range(2000)]
    occluder = occlusion.Occluder()
    serial = occluder.cull(rects)

    results = []
    errors = []

    def worker():
        try:
            results.append(occluder.cull(rects))
        except Exception as exc:  # pragma: no cover - 失败时暴露
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(results) == 8
    for r in results:
        assert r == serial
    # 统计在并发下累计一致。
    st = occluder.stats()
    assert st["total"] == 2000 * 9
    assert st["kept"] == len(serial) * 9
    assert st["culled"] == (2000 - len(serial)) * 9


def test_stats_accumulate_across_calls():
    o = occlusion.Occluder()
    o.cull([(0, 0, 10, 10), (1, 1, 2, 2)])
    o.cull([(0, 0, 1, 1)])
    st = o.stats()
    assert st["total"] == 3
    assert st["culled"] == 1
    assert st["kept"] == 2


def test_input_not_mutated():
    rects = [(0, 0, 10, 10), (1, 1, 2, 2), (20, 20, 5, 5)]
    snapshot = list(rects)
    occlusion.Occluder().cull(rects)
    assert rects == snapshot


def test_perf_5000_under_budget():
    import time
    rng = random.Random(5)
    rects = [(rng.randint(0, 10000), rng.randint(0, 10000),
              rng.randint(1, 100), rng.randint(1, 100))
             for _ in range(5000)]
    occlusion.Occluder().cull(rects)  # warmup
    t0 = time.perf_counter()
    out = occlusion.Occluder().cull(rects)
    dt_ms = (time.perf_counter() - t0) * 1000
    assert dt_ms < 200
    assert set(out) == _brute(rects)
