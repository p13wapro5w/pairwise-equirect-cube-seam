# pairwise-equirect-cube-seam

## 领域

游戏、图形与多媒体

## 主题

全景投影切换接缝

## 说明

映射版本、传感器对齐与面级完成标记。

初始状态下 `media/equirect_cube_seam.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
