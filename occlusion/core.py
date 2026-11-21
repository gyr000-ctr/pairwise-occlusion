"""2D 矩形遮挡剔除核心 API。

语义（与两两包含测试完全一致）：矩形 r 被剔除，当且仅当输入中存在
另一个矩形 r'，满足 r' 完全覆盖 r（边界相切也算覆盖）。部分重叠保留。

实现：矩形规范为 (x0, y0, x1, y1) 后按 (x0 升序, y0 升序, x1 降序,
y1 降序) 排序扫描，x0'<=x0 由扫描序天然满足，问题降为三维支配：
y0'<=y0 且 y1'>=y1 且 x1'>=x1。

数据结构为嵌套 Fenwick（树套树），规模相关数据全部放在紧凑 array
中（C 级连续存储，无逐元素 Python 对象）：
- y0、y1 分别压缩成整数秩；外层 Fenwick 建在 y0 秩上；
- 每个外层节点离线收集更新路径上所有矩形的 y1 秩，排序去重后存入
  定长紧凑数组，配等长的 Fenwick 最大值数组（存 x1，array('d')）；
- 为支持 "y1'>=y1 的 x1 最大值" 后缀查询，内层下标按 y1 秩逆序
  映射，后缀最大值即 Fenwick 前缀最大值；
- 在线扫描只做点更新（存活矩形）与前缀查询，均为 O(log^2 n)。
坐标存 array('d')、索引/秩存 array('q')、存活与重复标志存
bytearray；去重靠排序后相邻比较，秩靠 bisect 求得，主流程不保留
几何->计数或值->秩字典。被包含者不入树（包含关系可传递，其遮挡者
足以覆盖它能覆盖的一切）。
"""

from array import array
from bisect import bisect_left
import math
import threading


class OcclusionError(ValueError):
    """遮挡剔除相关错误的基类（仍是 ValueError，兼容既有调用方）。"""


class NonFiniteCoordinateError(OcclusionError):
    """矩形坐标中出现 NaN 或 ±inf（含运算后溢出为 inf 的边界）。"""


class DegenerateRectError(OcclusionError):
    """退化矩形：宽度或高度不大于 0。"""


def _validate(rect):
    """校验单个 (x, y, w, h) 矩形，返回规范形式 (x0, y0, x1, y1)。"""
    if not isinstance(rect, (tuple, list)) or len(rect) != 4:
        raise TypeError("rect must be a 4-element (x, y, w, h) sequence")
    for v in rect:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise TypeError("rect coordinates must be real numbers")
        if not math.isfinite(v):
            raise NonFiniteCoordinateError(
                "non-finite coordinate in rect %r" % (rect,))
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        raise DegenerateRectError(
            "degenerate rect (w=%r, h=%r) in %r" % (w, h, rect))
    x1 = x + w
    y1 = y + h
    if not math.isfinite(x1) or not math.isfinite(y1):
        raise NonFiniteCoordinateError(
            "rect %r produces non-finite bounds" % (rect,))
    return (x, y, x1, y1)


class Occluder:
    def __init__(self):
        self._lock = threading.Lock()
        self._stats = {"total": 0, "culled": 0, "kept": 0}

    def cull(self, rects):
        """返回保留矩形（按输入顺序），剔除被完全遮挡的矩形。"""
        with self._lock:
            rects = list(rects)
            n = len(rects)
            if n == 0:
                return []

            # 校验并紧凑打包：xs[4i..4i+3] = x0,y0,x1,y1。
            xs = array("d")
            for r in rects:
                x, y, x1, y1 = _validate(r)
                xs.append(x)
                xs.append(y)
                xs.append(x1)
                xs.append(y1)

            # 整数索引排序（key 临时元组仅在排序期间存活）。
            order = list(range(n))
            order.sort(key=lambda i: (xs[4 * i], xs[4 * i + 1],
                                      -xs[4 * i + 2], -xs[4 * i + 3]))

            # 去重为唯一扫描序列：相邻比较，记录每个输入位置对应的
            # 唯一下标，紧凑存扫描所需的三个坐标数组（x0 仅用于排序，
            # 排序后不再需要，故不保留）。
            uy0 = array("d")
            ux1 = array("d")
            uy1 = array("d")
            pos_of = array("q", [-1]) * n
            last_key = None
            k = 0
            for i in order:
                b = (xs[4 * i], xs[4 * i + 1],
                     xs[4 * i + 2], xs[4 * i + 3])
                if b != last_key:
                    last_key = b
                    uy0.append(b[1])
                    ux1.append(b[2])
                    uy1.append(b[3])
                    k += 1
                pos_of[i] = k - 1
            del order, last_key, xs  # 排序与规范临时区立即释放。

            # 重复几何：同一唯一下标被多个输入位置指向。
            seen = bytearray(k)
            dup_flag = bytearray(k)
            for i in range(n):
                p = pos_of[i]
                if seen[p]:
                    dup_flag[p] = 1
                else:
                    seen[p] = 1
            del seen

            # y0、y1 压缩值（紧凑、升序）；秩用 bisect 求，不建字典。
            y0_vals = array("d", sorted(set(uy0)))
            y1_vals = array("d", sorted(set(uy1)))
            m = len(y0_vals)

            # 离线分发：每外层 Fenwick 节点收集更新路径上的 y1 秩。
            buckets = [array("q") for _ in range(m + 1)]
            for idx in range(k):
                i = bisect_left(y0_vals, uy0[idx]) + 1
                r1 = bisect_left(y1_vals, uy1[idx])
                while i <= m:
                    buckets[i].append(r1)
                    i += i & -i

            # 每节点定长紧凑坐标（升序 y1 秩）+ Fenwick 最大值数组。
            coords = [None] * (m + 1)
            bits = [None] * (m + 1)
            NEG = -math.inf
            for i in range(1, m + 1):
                by = buckets[i]
                if not by:
                    continue
                coords[i] = array("q", sorted(set(by)))
                bits[i] = array("d", [NEG]) * (len(coords[i]) + 1)
            del buckets

            alive = bytearray(k)

            for idx in range(k):
                y0 = uy0[idx]
                x1 = ux1[idx]
                y1 = uy1[idx]
                rank0 = bisect_left(y0_vals, y0) + 1
                r1_global = bisect_left(y1_vals, y1)

                # 查询 y0'<=y0 前缀节点上 y1'>=y1 的后缀最大 x1。
                occluded = False
                i = rank0
                while i:
                    c = coords[i]
                    if c is not None:
                        kc = len(c)
                        p = bisect_left(c, r1_global)
                        lim = kc - p  # 逆序 Fenwick 前缀长度
                        bit = bits[i]
                        j = lim
                        best = NEG
                        while j:
                            v = bit[j]
                            if v > best:
                                best = v
                            j -= j & -j
                        if best >= x1:
                            occluded = True
                            break
                    i -= i & -i

                if occluded:
                    continue

                alive[idx] = 1

                # 点更新：外层更新路径上的内层 Fenwick 写最大值。
                i = rank0
                while i <= m:
                    c = coords[i]
                    kc = len(c)
                    p0 = bisect_left(c, r1_global)
                    pos = kc - p0  # 逆序 1 基位置
                    bit = bits[i]
                    j = pos
                    while j <= kc:
                        if x1 > bit[j]:
                            bit[j] = x1
                        j += j & -j
                    i += i & -i

            # 按输入顺序产出：唯一下标必须存活且该几何不重复。
            result = []
            for i in range(n):
                p = pos_of[i]
                if alive[p] and not dup_flag[p]:
                    result.append(rects[i])

            kept = len(result)
            self._stats["total"] += n
            self._stats["culled"] += n - kept
            self._stats["kept"] += kept
            return result

    def stats(self):
        """返回累计统计快照。"""
        with self._lock:
            return dict(self._stats)
