"""occlusion 核心 API（初始未实现）。"""


class Occluder:
    def __init__(self):
        """构造"""
        raise NotImplementedError

    def cull(self, rects):
        """返回保留矩形"""
        raise NotImplementedError

    def stats(self):
        """统计"""
        raise NotImplementedError
