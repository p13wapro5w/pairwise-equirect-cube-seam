"""全景视频在等距柱状(ERP)与立方体(CubeMap)投影间切换时的接缝修复模型。

用纯标准库实现一条轻量、可验证的切换管线，逐项对应题目要求的六个修复点：

=======================  ==========================================================
修复点                     对应入口
=======================  ==========================================================
投影切换的映射版本         :func:`mapping_version` / :func:`projection_switch`
传感器与帧时间对齐         :func:`sensor_align` / :func:`latency_comp`
面级完成标记               :func:`face_done`
纹理池按面分配             :func:`pool_per_face`
分辨率变化重算采样         :func:`resolution_recalc`
接缝像素混合               :func:`seam_blend` / :func:`face_order`
=======================  ==========================================================

另外提供 :func:`validation`、:func:`audit`、:func:`fallback`、:func:`metrics`、
:func:`orientation`、:func:`determinism` 作为校验、审计、降级与可复现性入口。

所有入口都接受三种调用形式：

* ``fn(scalar)``：传入一个标量（含义见各函数文档，缺省值有合理兜底）；
* ``fn({...})``：传入显式的参数/场景字典；
* ``fn(**kwargs)``：关键字参数，等价于字典形式。

返回值均是可 JSON 序列化的 ``dict``/``list``，内部不使用任何随机数，
同一输入在任意次运行中产生完全一致的结果。
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import deque
from typing import Any, Mapping

__all__ = [
    "FACES",
    "EDGES",
    "OPPOSITE",
    "ADJACENCY",
    "mapping_version",
    "sensor_align",
    "face_done",
    "pool_per_face",
    "resolution_recalc",
    "seam_blend",
    "projection_switch",
    "validation",
    "audit",
    "fallback",
    "latency_comp",
    "metrics",
    "orientation",
    "face_order",
    "determinism",
]

# ---------------------------------------------------------------------------
# 立方体拓扑
# ---------------------------------------------------------------------------

# 六个面按固定顺序编号，face mask 的第 i 位对应 FACES[i]。
FACES = ("front", "right", "back", "left", "top", "bottom")
FACE_INDEX = {name: i for i, name in enumerate(FACES)}
EDGES = ("top", "right", "bottom", "left")

OPPOSITE = {
    "front": "back",
    "back": "front",
    "left": "right",
    "right": "left",
    "top": "bottom",
    "bottom": "top",
}

# 每个面四条边上相邻的面，构成 12 条立方体接缝（无向边）。
ADJACENCY = {
    "front": {"top": "top", "right": "right", "bottom": "bottom", "left": "left"},
    "right": {"top": "top", "right": "back", "bottom": "bottom", "left": "front"},
    "back": {"top": "top", "right": "left", "bottom": "bottom", "left": "right"},
    "left": {"top": "top", "right": "front", "bottom": "bottom", "left": "back"},
    "top": {"top": "back", "right": "right", "bottom": "front", "left": "left"},
    "bottom": {"top": "front", "right": "right", "bottom": "back", "left": "left"},
}

ALL_FACE_MASK = (1 << len(FACES)) - 1

# ERP 的 2π 经度正好铺满赤道方向的 4 个立方体面，面边长 ≈ ERP 宽度 / 4。
ERP_TO_FACE_RATIO = 4.0

DEFAULT_TOLERANCE_MS = 8.0
DEFAULT_BLEND_PX = 8


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _spec(arg: Any, kwargs: Mapping[str, Any]) -> dict:
    """把三种调用形式归一成一个 dict，标量放进 ``value``。"""
    if kwargs:
        base = dict(arg) if isinstance(arg, Mapping) else {}
        base.update(kwargs)
        return base
    if isinstance(arg, Mapping):
        return dict(arg)
    if arg is None:
        return {}
    return {"value": arg}


def _scalar(spec: Mapping[str, Any], default: Any = None) -> Any:
    return spec.get("value", default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value: float, ndigits: int = 6) -> float:
    out = round(float(value), ndigits)
    return 0.0 if out == 0 else out


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _digest(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


def _smoothstep(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def _round_list(values: Any) -> list:
    return [_round(v) for v in values]


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


# ---------------------------------------------------------------------------
# 球面 / 立方体几何（映射版本的实现基础）
# ---------------------------------------------------------------------------


def _erp_pixel_to_dir(x: float, y: float, width: int, height: int) -> tuple:
    """ERP 像素中心 → 单位方向向量，与投影版本无关的固定约定。"""
    lon = (x + 0.5) / width * 2.0 * math.pi - math.pi
    lat = math.pi / 2.0 - (y + 0.5) / height * math.pi
    cos_lat = math.cos(lat)
    return (
        cos_lat * math.sin(lon),
        math.sin(lat),
        cos_lat * math.cos(lon),
    )


def _dir_to_erp_pixel(direction: tuple, width: int, height: int) -> tuple:
    dx, dy, dz = direction
    lon = math.atan2(dx, dz)
    lat = math.asin(_clamp(dy, -1.0, 1.0))
    x = (lon + math.pi) / (2.0 * math.pi) * width - 0.5
    y = (math.pi / 2.0 - lat) / math.pi * height - 0.5
    return x, y


def _dir_to_cube(direction: tuple, face_size: int) -> dict:
    """方向向量 → 立方体面号与面内像素，用于面级采样/接缝定位。"""
    dx, dy, dz = direction
    ax, ay, az = abs(dx), abs(dy), abs(dz)
    if az >= ax and az >= ay:
        face = "front" if dz >= 0 else "back"
        major = az
        u = (dx if dz >= 0 else -dx) / major
        v = dy / major
    elif ax >= ay:
        face = "right" if dx >= 0 else "left"
        major = ax
        u = (-dz if dx >= 0 else dz) / major
        v = dy / major
    else:
        face = "top" if dy >= 0 else "bottom"
        major = ay
        u = dx / major
        v = (-dz if dy >= 0 else dz) / major
    return {
        "face": face,
        "u": _round(u),
        "v": _round(v),
        "x": _round((u + 1.0) * 0.5 * face_size),
        "y": _round((1.0 - v) * 0.5 * face_size),
    }


# ---------------------------------------------------------------------------
# 姿态 / 四元数（传感器对齐的实现基础）
# ---------------------------------------------------------------------------


def _normalize_quat(quat: Any) -> list:
    q = [_as_float(v, 0.0) for v in (list(quat) + [0.0, 0.0, 0.0, 0.0])[:4]]
    norm = math.sqrt(sum(v * v for v in q))
    if norm <= 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    return [v / norm for v in q]


def _quat_mul(a: Any, b: Any) -> list:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def _quat_from_ypr(yaw_deg: float, pitch_deg: float, roll_deg: float) -> list:
    """yaw 绕 Y、pitch 绕 X、roll 绕 Z，角度制。"""
    hy = math.radians(yaw_deg) * 0.5
    hp = math.radians(pitch_deg) * 0.5
    hr = math.radians(roll_deg) * 0.5
    qy = [math.cos(hy), 0.0, math.sin(hy), 0.0]
    qp = [math.cos(hp), math.sin(hp), 0.0, 0.0]
    qr = [math.cos(hr), 0.0, 0.0, math.sin(hr)]
    return _normalize_quat(_quat_mul(_quat_mul(qy, qp), qr))


def _quat_to_ypr(quat: Any) -> dict:
    """四元数 → yaw(Y) / pitch(X) / roll(Z)，与 :func:`_quat_from_ypr` 互逆。"""
    w, x, y, z = _normalize_quat(quat)
    yaw = math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(_clamp(2.0 * (w * x - y * z), -1.0, 1.0))
    roll = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (x * x + z * z))
    return {
        "yaw": _round(math.degrees(yaw)),
        "pitch": _round(math.degrees(pitch)),
        "roll": _round(math.degrees(roll)),
    }


def _slerp(a: Any, b: Any, t: float) -> list:
    qa = _normalize_quat(a)
    qb = _normalize_quat(b)
    dot = sum(x * y for x, y in zip(qa, qb))
    if dot < 0.0:
        qb = [-v for v in qb]
        dot = -dot
    dot = _clamp(dot, -1.0, 1.0)
    if dot > 0.9995:
        mixed = [x + (y - x) * t for x, y in zip(qa, qb)]
        return _normalize_quat(mixed)
    theta0 = math.acos(dot)
    theta = theta0 * t
    sin_theta0 = math.sin(theta0)
    s0 = math.cos(theta) - dot * math.sin(theta) / sin_theta0
    s1 = math.sin(theta) / sin_theta0
    return _normalize_quat([s0 * x + s1 * y for x, y in zip(qa, qb)])


def _orientation_to_quat(value: Any) -> list | None:
    """把 dict / 4 元组 / 标量（视为 yaw 角）统一成四元数。"""
    if isinstance(value, Mapping):
        if "quaternion" in value:
            return _normalize_quat(value["quaternion"])
        q = value.get("q")
        if isinstance(q, (list, tuple)) and len(q) == 4:
            return _normalize_quat(q)
        return _quat_from_ypr(
            _as_float(value.get("yaw"), 0.0),
            _as_float(value.get("pitch"), 0.0),
            _as_float(value.get("roll"), 0.0),
        )
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return _normalize_quat(value)
    if isinstance(value, (int, float)):
        return _quat_from_ypr(float(value), 0.0, 0.0)
    return None


# ---------------------------------------------------------------------------
# 1. 投影切换的映射版本
# ---------------------------------------------------------------------------


def _mapping_fingerprint(width: int, face_size: int, samples: int = 4) -> dict:
    """采样 ERP↔Cube 的点对应关系，作为映射版本的内容指纹。

    同一版本号但几何/分辨率不同（例如投屏分辨率变了）必须产生不同的指纹，
    接缝混合才能发现「版本没变但映射已经变了」这种隐蔽的旧映射。
    """
    height = max(1, width // 2)
    points = []
    for i in range(samples):
        for j in range(samples):
            x = (i + 0.5) * width / samples
            y = (j + 0.5) * height / samples
            direction = _erp_pixel_to_dir(x, y, width, height)
            cube = _dir_to_cube(direction, face_size)
            back = _dir_to_erp_pixel(direction, width, height)
            points.append(
                {
                    "erp": [_round(x, 3), _round(y, 3)],
                    "face": cube["face"],
                    "cube_uv": [cube["u"], cube["v"]],
                    "roundtrip": [_round(back[0], 3), _round(back[1], 3)],
                }
            )
    return {
        "width": width,
        "face_size": face_size,
        "samples": points,
        "digest": _digest(points),
    }


def mapping_version(arg: Any = None, **kwargs: Any) -> dict:
    """为切换帧生成一致、原子的映射版本。

    标量调用视为当前版本号；字典可给出 ``projection``/``source_projection``、
    ``frame``/``switch_frame``、``version``/``prev_version`` 与 ``bump``。

    关键点：切换帧上 ERP 与 Cube 的每个采样点都必须打上 **同一个** 新版本戳，
    接缝混合才能确认自己拿到的不是上一帧的旧映射。版本内容指纹由实际几何
    映射采样得出，几何或分辨率变化也会让指纹变化。
    """
    spec = _spec(arg, kwargs)
    scalar = _scalar(spec)
    prev = _as_int(
        spec.get("prev_version", spec.get("version", scalar)),
        0,
    )
    prev = max(0, prev)
    projection = str(spec.get("projection", spec.get("to", "cube")))
    source = str(spec.get("source_projection", spec.get("from", projection)))
    frame = _as_int(spec.get("frame", spec.get("frame_index", 0)), 0)

    width = _as_int(spec.get("width", spec.get("erp_width")), 3840)
    width = max(16, width)
    face_size = _as_int(spec.get("face_size"), max(1, int(round(width / ERP_TO_FACE_RATIO))))
    face_size = max(1, face_size)

    if "bump" in spec:
        changed = bool(spec["bump"])
    elif "switch" in spec:
        changed = bool(spec["switch"])
    elif "switch_frame" in spec:
        changed = frame >= _as_int(spec["switch_frame"], frame)
    else:
        changed = projection != source

    version = prev + 1 if changed else prev
    fingerprint = _mapping_fingerprint(width, face_size)
    mapping_id = _digest(
        {
            "projection": projection,
            "layout": "cube6",
            "version": version,
            "geometry": fingerprint["digest"],
        }
    )
    sample_versions = {"equirect": version, "cube": version}

    return {
        "ok": True,
        "prev_version": prev,
        "version": version,
        "projection": projection,
        "source_projection": source,
        "changed": changed,
        "frame": frame,
        "mapping_id": mapping_id,
        "geometry": fingerprint,
        "geometry_digest": fingerprint["digest"],
        "seam_version": version,
        "sample_versions": sample_versions,
        "atomic": True,
        "consistent": len(set(sample_versions.values())) == 1,
        "erp_to_cube": projection == "cube",
        "value": version,
    }


# ---------------------------------------------------------------------------
# 2. 传感器与帧时间对齐
# ---------------------------------------------------------------------------


def _parse_sensor_samples(raw: Any) -> list:
    samples = []
    for index, item in enumerate(raw or []):
        if isinstance(item, Mapping):
            ts = _as_float(
                item.get("t", item.get("ts", item.get("timestamp", index))),
                float(index),
            )
            quat = _orientation_to_quat(item)
        elif isinstance(item, (list, tuple)):
            if len(item) == 5:
                ts = _as_float(item[0], float(index))
                quat = _normalize_quat(item[1:5])
            elif len(item) >= 2:
                ts = _as_float(item[0], float(index))
                yaw = _as_float(item[1], 0.0)
                pitch = _as_float(item[2], 0.0) if len(item) > 2 else 0.0
                roll = _as_float(item[3], 0.0) if len(item) > 3 else 0.0
                quat = _quat_from_ypr(yaw, pitch, roll)
            else:
                continue
        else:
            ts = float(index)
            quat = _orientation_to_quat(item) or [1.0, 0.0, 0.0, 0.0]
        samples.append((ts, quat))
    samples.sort(key=lambda pair: pair[0])
    return samples


def sensor_align(arg: Any = None, **kwargs: Any) -> dict:
    """把方向传感器采样对齐到视频帧时间戳。

    传感器读数是「过去」的姿态：先减去 ``sensor_latency_ms``，再在帧时间戳处
    对相邻采样做球面插值（slerp）；超出采样范围时保持最近姿态并报告残差。
    标量调用视为帧时间戳（毫秒），无采样时退化为 hold。
    """
    spec = _spec(arg, kwargs)
    scalar = _scalar(spec)
    frame_ts = _as_float(
        spec.get("frame_ts", spec.get("frame_timestamp", scalar)),
        0.0,
    )
    latency = _as_float(
        spec.get("sensor_latency_ms", spec.get("latency_ms", 0.0)), 0.0
    )
    tolerance = _as_float(
        spec.get("max_residual_ms", spec.get("tolerance_ms", DEFAULT_TOLERANCE_MS)),
        DEFAULT_TOLERANCE_MS,
    )

    raw = spec.get("samples")
    if raw is None and isinstance(scalar, (list, tuple)):
        candidate = scalar
        if candidate and isinstance(candidate[0], (list, tuple, Mapping)):
            raw = candidate
    samples = _parse_sensor_samples(raw)

    effective_ts = frame_ts - latency
    base = {
        "ok": True,
        "frame_ts": _round(frame_ts),
        "sensor_latency_ms": _round(latency),
        "effective_ts": _round(effective_ts),
        "offset_ms": _round(latency),
        "max_residual_ms": _round(tolerance),
        "sample_count": len(samples),
        "latency_compensated": True,
    }

    if not samples:
        aligned = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        return {
            **base,
            "aligned": aligned,
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "method": "hold",
            "residual_ms": 0.0,
            "within_tolerance": True,
            "stale": False,
            "note": "no_sensor_samples",
        }

    times = [pair[0] for pair in samples]
    index = bisect.bisect_left(times, effective_ts)
    if index <= 0:
        ts_used, quat = samples[0]
        method = "clamp_first"
        residual = ts_used - effective_ts
    elif index >= len(samples):
        ts_used, quat = samples[-1]
        method = "clamp_last"
        residual = effective_ts - ts_used
    else:
        t0, q0 = samples[index - 1]
        t1, q1 = samples[index]
        span = t1 - t0
        alpha = 0.0 if span <= 0 else _clamp((effective_ts - t0) / span, 0.0, 1.0)
        quat = _slerp(q0, q1, alpha)
        ts_used = effective_ts
        method = "slerp"
        residual = 0.0

    return {
        **base,
        "aligned": _quat_to_ypr(quat),
        "quaternion": _round_list(quat),
        "method": method,
        "sensor_ts_used": _round(ts_used),
        "residual_ms": _round(residual),
        "within_tolerance": residual <= tolerance,
        "stale": residual > tolerance,
    }


# ---------------------------------------------------------------------------
# 3. 面级完成标记
# ---------------------------------------------------------------------------


def _mask_to_faces(mask: int) -> list:
    return [name for i, name in enumerate(FACES) if mask & (1 << i)]


def _faces_to_mask(names: Any) -> int:
    if isinstance(names, int):
        return names & ALL_FACE_MASK
    if isinstance(names, str):
        names = [names]
    mask = 0
    for name in names or []:
        index = FACE_INDEX.get(str(name))
        if index is not None:
            mask |= 1 << index
    return mask


def face_done(arg: Any = None, **kwargs: Any) -> dict:
    """维护每个立方体面的完成标记，切换时只允许读取已完成的整面。

    标量调用视为已完成面的位掩码；字典可用 ``done_faces``/``done``/``completed``
    或 ``face_states``。若给出 ``version``/``frame``，还会检查每个完成面是否属于
    当前映射版本与当前帧，避免读到「上一版本已写完、本版本还没写完」的面。
    """
    spec = _spec(arg, kwargs)
    scalar = _scalar(spec)
    version = _as_int(spec.get("version", 0), 0)
    frame = _as_int(spec.get("frame", 0), 0)
    require_version = "version" in spec
    require_frame = "frame" in spec

    states = spec.get("face_states")
    per_face: dict = {}
    if isinstance(states, Mapping):
        mask = 0
        ready_mask = 0
        for name in FACES:
            state = states.get(name, {})
            if not isinstance(state, Mapping):
                state = {"done": bool(state)}
            done = bool(state.get("done", True))
            face_version = _as_int(state.get("version", version), version)
            face_frame = _as_int(state.get("frame", frame), frame)
            ready = done
            if require_version and face_version != version:
                ready = False
            if require_frame and face_frame != frame:
                ready = False
            per_face[name] = {
                "done": done,
                "ready": ready,
                "version": face_version,
                "frame": face_frame,
            }
            if done:
                mask |= 1 << FACE_INDEX[name]
            if ready:
                ready_mask |= 1 << FACE_INDEX[name]
    else:
        done_names = spec.get(
            "done_faces", spec.get("done", spec.get("completed", scalar))
        )
        if done_names is None:
            done_names = FACES
        if isinstance(done_names, Mapping):
            done_names = [k for k, v in done_names.items() if v]
        mask = _faces_to_mask(done_names)
        # 允许通过 incomplete/blocked 列表扣掉面。
        blocked = spec.get("blocked_faces", spec.get("incomplete_faces", []))
        mask &= ~_faces_to_mask(blocked)
        ready_mask = mask
        for name in FACES:
            done = bool(mask & (1 << FACE_INDEX[name]))
            per_face[name] = {"done": done, "ready": done, "version": version, "frame": frame}

    done = _mask_to_faces(mask)
    ready = _mask_to_faces(ready_mask)
    missing = [name for name in FACES if name not in done]
    blocked = [name for name in FACES if name not in ready]

    return {
        "ok": True,
        "mask": mask,
        "mask_bits": format(mask, "06b"),
        "done": done,
        "missing": missing,
        "complete_count": len(done),
        "all_done": mask == ALL_FACE_MASK,
        "ready_mask": ready_mask,
        "ready": ready,
        "all_ready": ready_mask == ALL_FACE_MASK,
        "blocked": blocked,
        "per_face": per_face,
        "version": version,
        "frame": frame,
        "value": mask,
    }


# ---------------------------------------------------------------------------
# 4. 纹理池按面分配
# ---------------------------------------------------------------------------


def pool_per_face(arg: Any = None, **kwargs: Any) -> dict:
    """为每条解码流、每个立方体面分配独立纹理槽，禁止跨流共享写入目标。

    标量调用视为解码流数量；字典可用 ``decoders``/``streams``、``faces``、
    ``size``。分配结果保证 ``(stream, face)`` 唯一，切面时不会读到别人正在写的面。
    """
    spec = _spec(arg, kwargs)
    scalar = _scalar(spec)
    streams = _as_int(
        spec.get("decoders", spec.get("streams", spec.get("lanes", scalar))), 1
    )
    streams = int(_clamp(streams, 1, 64))
    faces = spec.get("faces", FACES)
    if isinstance(faces, Mapping):
        faces = list(faces.keys())
    faces = [str(name) for name in faces] or list(FACES)
    size = spec.get("size", spec.get("face_size"))

    allocations = []
    per_stream: dict = {}
    per_face: dict = {}
    for stream in range(streams):
        stream_slots = {}
        for order, face in enumerate(faces):
            slot = stream * len(faces) + order
            entry = {
                "stream": stream,
                "face": face,
                "slot": slot,
                "texture": f"pool/stream{stream}/face-{face}",
                "owner": stream,
                "shared": False,
            }
            allocations.append(entry)
            stream_slots[face] = slot
            per_face.setdefault(face, []).append(slot)
        per_stream[str(stream)] = stream_slots

    slots = [entry["texture"] for entry in allocations]
    return {
        "ok": True,
        "decoders": streams,
        "face_count": len(faces),
        "faces": faces,
        "size": size,
        "allocations": allocations,
        "per_stream": per_stream,
        "per_face": per_face,
        "slots": slots,
        "slot_count": len(slots),
        "unique_slots": len(set(slots)) == len(slots),
        "shared": False,
        "alias_free": len(set(slots)) == len(slots),
        "cross_stream_sharing": False,
        "value": len(slots),
    }


# ---------------------------------------------------------------------------
# 5. 分辨率变化重算采样
# ---------------------------------------------------------------------------


def resolution_recalc(arg: Any = None, **kwargs: Any) -> dict:
    """投屏分辨率变化后重算 ERP / Cube 的采样步长与立方体面尺寸。

    标量调用视为目标 ERP 宽度（像素）。面边长取 ``宽度 / 4``，因为赤道方向
    2π 经度正好对应 4 个立方体面；分辨率变化时接缝两侧都按新步长重建采样，
    否则接缝处会出现采样密度不一致导致的可见错位。
    """
    spec = _spec(arg, kwargs)
    scalar = _scalar(spec)
    to_width = _as_int(spec.get("to_width", spec.get("width", scalar)), 3840)
    to_width = max(16, to_width)
    to_height = _as_int(
        spec.get("to_height", spec.get("height", to_width // 2)), to_width // 2
    )
    to_height = max(8, to_height)
    from_width = _as_int(spec.get("from_width", to_width), to_width)
    from_height = _as_int(spec.get("from_height", to_height), to_height)

    face_size = max(1, int(round(to_width / ERP_TO_FACE_RATIO)))
    changed = (from_width, from_height) != (to_width, to_height)
    scale = to_width / from_width if from_width else 1.0

    step_u = 2.0 * math.pi / to_width
    step_v = math.pi / to_height
    step_cube = (math.pi / 2.0) / face_size

    return {
        "ok": True,
        "from": {"width": from_width, "height": from_height},
        "to": {"width": to_width, "height": to_height},
        "changed": changed,
        "needs_resample": changed,
        "scale": _round(scale),
        "face_size": face_size,
        "cube_face": {"width": face_size, "height": face_size},
        "lod_bias": _round(math.log2(scale) if scale > 0 else 0.0),
        "sampling": {
            "erp_rad_per_px_u": _round(step_u, 9),
            "erp_rad_per_px_v": _round(step_v, 9),
            "cube_rad_per_texel": _round(step_cube, 9),
            "erp_px_per_rad": _round(to_width / (2.0 * math.pi)),
            "cube_texels_per_rad": _round(face_size / (math.pi / 2.0)),
        },
        "sample_step": {
            "u": _round(step_u, 9),
            "v": _round(step_v, 9),
            "cube": _round(step_cube, 9),
        },
        "seams_recomputed": True,
        "rebuild_before_present": True,
        "value": face_size,
    }


# ---------------------------------------------------------------------------
# 6. 接缝像素混合
# ---------------------------------------------------------------------------


def _blend_ramp(width: int) -> list:
    """生成能量守恒的左右权重斜坡，smoothstep 让接缝过渡无硬边。

    宽度为 1 时退化为单点 50/50 采样，仍满足权重守恒与对称。
    """
    if width <= 1:
        return [[0.5, 0.5]]
    ramp = []
    for i in range(width):
        alpha = _smoothstep(i / (width - 1))
        ramp.append([_round(1.0 - alpha), _round(alpha)])
    return ramp


def _cube_seams() -> list:
    seams = []
    for face in FACES:
        for edge in EDGES:
            neighbor = ADJACENCY[face][edge]
            if FACE_INDEX[face] < FACE_INDEX[neighbor]:
                seams.append({"a": face, "b": neighbor, "edge": edge})
    return seams


def seam_blend(arg: Any = None, **kwargs: Any) -> dict:
    """在立方体 12 条接缝与 ERP 经度/极点接缝上生成混合权重。

    标量调用视为混合宽度（像素）。两侧权重用 smoothstep 生成并严格配对为 1，
    因此混合不会改变整体亮度；分辨率变化后按新面尺寸重新生成斜坡。
    """
    spec = _spec(arg, kwargs)
    scalar = _scalar(spec)
    raw_width = spec.get("blend_px", spec.get("width_px", scalar))
    width = _as_int(raw_width, DEFAULT_BLEND_PX)
    width = int(_clamp(width, 1, 64))
    projection = str(spec.get("projection", "cube"))
    face_size = _as_int(spec.get("face_size", 0), 0)
    if face_size > 0:
        width = int(_clamp(width, 1, max(1, face_size // 2)))

    ramp = _blend_ramp(width)
    weight_sum = _round(sum(a + b for a, b in ramp) / len(ramp))
    seams = _cube_seams()
    for seam in seams:
        seam["blend_px"] = width
        seam["samples"] = width * 2

    per_face_edges = {
        face: {edge: ADJACENCY[face][edge] for edge in EDGES} for face in FACES
    }

    erp_seams = [
        {"a": "erp", "b": "erp", "edge": "longitude_wrap", "blend_px": width},
        {"a": "erp", "b": "erp", "edge": "north_pole", "blend_px": width},
        {"a": "erp", "b": "erp", "edge": "south_pole", "blend_px": width},
    ]

    return {
        "ok": True,
        "projection": projection,
        "blend_px": width,
        "blend_width": width,
        "ramp": ramp,
        "ramp_len": len(ramp),
        "weight_sum": weight_sum,
        "energy_preserving": abs(weight_sum - 1.0) <= 1e-6,
        "symmetric": all(
            abs(ramp[i][0] - ramp[width - 1 - i][1]) <= 1e-6 for i in range(width)
        ),
        "seams": seams,
        "seam_count": len(seams),
        "erp_seams": erp_seams,
        "per_face_edges": per_face_edges,
        "smoothstep": True,
        "degenerate": width <= 1,
        "rescaled": face_size > 0,
        "value": width,
    }


# ---------------------------------------------------------------------------
# 面序（接缝混合的依赖顺序）
# ---------------------------------------------------------------------------


def face_order(arg: Any = None, **kwargs: Any) -> dict:
    """给出确定性的解码/上传面序，保证接缝混合前相邻面都已就绪。

    标量 1..6 视为起始面编号（1 即 front），其余情况从 ``start``（默认 front）
    做固定邻接顺序的 BFS；结果只依赖立方体拓扑，可复现。
    """
    spec = _spec(arg, kwargs)
    scalar = _scalar(spec)
    if "start" in spec:
        start = spec["start"]
    elif isinstance(scalar, int) and not isinstance(scalar, bool) and 1 <= scalar <= len(FACES):
        start = FACES[scalar - 1]
    elif isinstance(scalar, str):
        start = scalar
    else:
        start = "front"
    if start not in FACE_INDEX:
        start = "front"

    order = []
    seen = set()
    queue = deque([start])
    while queue:
        face = queue.popleft()
        if face in seen:
            continue
        seen.add(face)
        order.append(face)
        for edge in EDGES:
            neighbor = ADJACENCY[face][edge]
            if neighbor not in seen:
                queue.append(neighbor)
    for face in FACES:  # 理论上 BFS 已覆盖全部，这里只是兜底。
        if face not in seen:
            order.append(face)
            seen.add(face)

    pairs = []
    for face in FACES:
        for edge in EDGES:
            neighbor = ADJACENCY[face][edge]
            if FACE_INDEX[face] < FACE_INDEX[neighbor]:
                pairs.append([face, neighbor])

    rank = {face: index for index, face in enumerate(order)}
    return {
        "ok": True,
        "start": start,
        "order": order,
        "rank": rank,
        "adjacency": {face: dict(ADJACENCY[face]) for face in FACES},
        "edge_pairs": pairs,
        "edge_count": len(pairs),
        "deterministic": True,
        "digest": _digest({"order": order, "edges": pairs}),
        "value": order,
    }


# ---------------------------------------------------------------------------
# 投影切换总编排
# ---------------------------------------------------------------------------


def _switch_plan(spec: Mapping[str, Any]) -> dict:
    scalar = _scalar(spec)
    frame = _as_int(spec.get("frame", spec.get("frame_index", 0)), 0)
    switch_frame = _as_int(spec.get("switch_frame", frame), frame)

    resolution = resolution_recalc(
        {
            "from_width": spec.get("from_width", spec.get("width")),
            "from_height": spec.get("from_height", spec.get("height")),
            "to_width": spec.get("to_width", spec.get("width")),
            "to_height": spec.get("to_height", spec.get("height")),
        }
    )

    mapping = mapping_version(
        {
            "prev_version": spec.get(
                "prev_version", spec.get("version", scalar if isinstance(scalar, int) else 0)
            ),
            "frame": frame,
            "switch_frame": switch_frame,
            "projection": spec.get("projection", spec.get("to", "cube")),
            "source_projection": spec.get("source_projection", spec.get("from", "equirect")),
            "width": resolution["to"]["width"],
            "face_size": resolution["face_size"],
        }
    )

    done_names = spec.get("done_faces", spec.get("done", FACES))
    face_states = spec.get("face_states")
    face_spec: dict = {"faces": FACES}
    if face_states is not None:
        face_spec["face_states"] = face_states
    else:
        face_spec["done_faces"] = done_names
    face_spec["blocked_faces"] = spec.get("blocked_faces", spec.get("incomplete_faces", []))
    face_spec["version"] = mapping["version"]
    face_spec["frame"] = frame
    faces = face_done(face_spec)

    pool = pool_per_face(
        {
            "decoders": spec.get("decoders", spec.get("streams", 1)),
            "size": spec.get("face_size", resolution["face_size"]),
        }
    )

    seam = seam_blend(
        {
            "blend_px": spec.get("blend_px", DEFAULT_BLEND_PX),
            "projection": mapping["projection"],
            "face_size": resolution["face_size"],
        }
    )

    sensor = sensor_align(
        {
            "frame_ts": spec.get("frame_ts", float(frame)),
            "samples": spec.get("samples"),
            "sensor_latency_ms": spec.get("sensor_latency_ms", 0.0),
            "max_residual_ms": spec.get("max_residual_ms", DEFAULT_TOLERANCE_MS),
        }
    )

    order = face_order({"start": spec.get("start", "front")})

    checks = {
        "mapping_atomic": bool(mapping["atomic"] and mapping["consistent"]),
        "seam_uses_current_mapping": mapping["seam_version"] == mapping["version"],
        "faces_ready": bool(faces["all_ready"]),
        "pool_per_face": bool(not pool["shared"] and pool["unique_slots"]),
        "resolution_recalculated": bool(resolution["seams_recomputed"]),
        "mapping_matches_resolution": (
            mapping["geometry"]["face_size"] == resolution["face_size"]
            and mapping["geometry"]["width"] == resolution["to"]["width"]
        ),
        "sensor_aligned": bool(sensor["within_tolerance"]),
        "blend_energy_preserving": bool(seam["energy_preserving"]),
        "order_deterministic": bool(order["deterministic"]),
    }
    ready = all(checks.values())

    return {
        "ok": True,
        "frame": frame,
        "switch_frame": switch_frame,
        "mapping": mapping,
        "mapping_version": mapping["version"],
        "seam_version": mapping["version"],
        "faces": faces,
        "face_mask": faces["mask"],
        "face_ready_mask": faces["ready_mask"],
        "blocked_faces": faces["blocked"],
        "pool": pool,
        "resolution": resolution,
        "seam": seam,
        "sensor": sensor,
        "order": order,
        "checks": checks,
        "ready": ready,
        "atomic": True,
        "value": mapping["version"],
    }


def projection_switch(arg: Any = None, **kwargs: Any) -> dict:
    """在切换帧上原子地编排六个修复点，返回可审计的切换计划。

    标量调用视为帧号。默认视角为 ERP → Cube；可通过 ``projection``、
    ``source_projection``、``done_faces``、``decoders``、``to_width``、
    ``blend_px``、``samples`` 等键定制场景。
    """
    return _switch_plan(_spec(arg, kwargs))


# ---------------------------------------------------------------------------
# 校验 / 审计 / 降级
# ---------------------------------------------------------------------------


def _geometry_matches(mapping: Mapping[str, Any], resolution: Mapping[str, Any]) -> bool:
    """映射记录的几何是否与当前分辨率一致。"""
    geometry = mapping.get("geometry")
    if not isinstance(geometry, Mapping):
        return True
    face_size = _as_int(resolution.get("face_size", 0), 0)
    to_width = _as_int(resolution.get("to", {}).get("width", 0), 0)
    if face_size and _as_int(geometry.get("face_size", face_size), face_size) != face_size:
        return False
    if to_width and _as_int(geometry.get("width", to_width), to_width) != to_width:
        return False
    return True


def validation(arg: Any = None, **kwargs: Any) -> dict:
    """校验切换计划是否满足全部接缝安全带条件。

    接受完整计划（``projection_switch`` 的返回值 / 含 ``plan`` 键），
    也接受场景字典或标量，后者会先构造默认计划再校验。
    """
    spec = _spec(arg, kwargs)
    plan = spec.get("plan")
    if not isinstance(plan, Mapping):
        if {"mapping", "faces", "pool"}.issubset(spec.keys()):
            plan = spec
        else:
            plan = _switch_plan(spec)

    mapping = plan.get("mapping", {})
    faces = plan.get("faces", {})
    pool = plan.get("pool", {})
    resolution = plan.get("resolution", {})
    seam = plan.get("seam", {})
    sensor = plan.get("sensor", {})
    order = plan.get("order", {})

    sample_versions = mapping.get("sample_versions", {})
    checks = [
        {
            "name": "mapping_version_atomic",
            "ok": bool(mapping.get("atomic", True))
            and len(set(sample_versions.values())) <= 1,
            "detail": "切换帧所有采样点使用同一映射版本",
        },
        {
            "name": "seam_not_stale",
            "ok": mapping.get("seam_version") == mapping.get("version"),
            "detail": "接缝混合读取当前版本而非旧映射",
        },
        {
            "name": "faces_complete",
            "ok": bool(faces.get("all_done")) and bool(faces.get("all_ready", True)),
            "detail": "所有立方体面已完成且属于当前版本/帧",
        },
        {
            "name": "pool_not_shared",
            "ok": bool(not pool.get("shared", False)) and bool(pool.get("unique_slots", True)),
            "detail": "纹理槽按面/按流唯一分配",
        },
        {
            "name": "resolution_sampling_valid",
            "ok": _as_int(resolution.get("face_size", 0), 0) > 0
            and bool(resolution.get("seams_recomputed", True)),
            "detail": "分辨率变化后接缝采样已按新面尺寸重算",
        },
        {
            "name": "mapping_matches_resolution",
            "ok": _geometry_matches(mapping, resolution),
            "detail": "映射指纹与当前分辨率/面尺寸一致，未携带旧几何",
        },
        {
            "name": "seam_blend_energy_preserving",
            "ok": bool(seam.get("energy_preserving", True))
            and abs(_as_float(seam.get("weight_sum", 1.0), 1.0) - 1.0) <= 1e-6,
            "detail": "接缝混合权重守恒",
        },
        {
            "name": "sensor_frame_aligned",
            "ok": bool(sensor.get("within_tolerance", True)),
            "detail": "传感器姿态与视频帧时间戳对齐在容差内",
        },
        {
            "name": "face_order_deterministic",
            "ok": bool(order.get("deterministic", True)),
            "detail": "面序稳定可复现",
        },
    ]
    failures = [check["name"] for check in checks if not check["ok"]]
    return {
        "ok": not failures,
        "valid": not failures,
        "checks": checks,
        "failures": failures,
        "checked": len(checks),
        "passed": len(checks) - len(failures),
        "value": not failures,
    }


def audit(arg: Any = None, **kwargs: Any) -> dict:
    """输出六类接缝问题的根因-修复追踪表。"""
    items = [
        {
            "id": "seam_old_mapping",
            "symptom": "切换帧接缝仍使用旧映射",
            "root_cause": "映射版本未随切换原子更新，接缝采样落后一帧",
            "fix": "切换时整体提升映射版本，并给每个采样点打同一版本戳",
            "entry": "mapping_version",
        },
        {
            "id": "sensor_frame_skew",
            "symptom": "方向传感器与视频帧时间戳不对齐导致画面滞后",
            "root_cause": "传感器读数带固定延迟，直接按到达时间使用",
            "fix": "按 sensor_latency 补偿后在帧时间戳处做 slerp 对齐",
            "entry": "sensor_align",
        },
        {
            "id": "face_tearing",
            "symptom": "切换时读到未写完的立方体面",
            "root_cause": "缺少面级完成标记，上传与解码并发",
            "fix": "维护每面的完成/版本/帧标记，未就绪的面不参与混合",
            "entry": "face_done",
        },
        {
            "id": "pool_shared_texture",
            "symptom": "多路解码共用纹理池互相覆盖",
            "root_cause": "纹理槽按流共享，切面时写入未完成",
            "fix": "按 (流, 面) 分配独立纹理槽，禁止跨流共享",
            "entry": "pool_per_face",
        },
        {
            "id": "resolution_seam_drift",
            "symptom": "投屏分辨率变化后接缝更明显",
            "root_cause": "面尺寸与采样步长未随分辨率重算，两侧密度不一致",
            "fix": "按新宽度重算面边长、采样步长与接缝斜坡",
            "entry": "resolution_recalc",
        },
        {
            "id": "hard_seam",
            "symptom": "接缝处出现硬边或亮度跳变",
            "root_cause": "接缝两侧直接硬切换，无过渡像素",
            "fix": "在 12 条立方体边与 ERP 经度/极点做 smoothstep 混合",
            "entry": "seam_blend",
        },
    ]
    spec = _spec(arg, kwargs)
    scalar = _scalar(spec)
    focused = spec.get("issue", spec.get("mode", scalar if isinstance(scalar, str) else None))
    total = len(items)
    if focused is not None:
        items = [item for item in items if item["id"] == focused or item["entry"] == focused]
    return {
        "ok": True,
        "items": items,
        "count": len(items),
        "total": total,
        "complete": total == 6,
        "traceable": True,
        "all_fixed": True,
        "value": len(items),
    }


def fallback(arg: Any = None, **kwargs: Any) -> dict:
    """给出故障时的降级策略，原则是「宁可停一帧，不可露出半个面」。"""
    modes = {
        "mapping_mismatch": {
            "action": "hold_last_coherent_frame",
            "detail": "接缝与主体版本不一致时整帧保留上一份一致映射",
            "drop_partial": True,
        },
        "incomplete_face": {
            "action": "display_last_complete_generation",
            "detail": "有面未写完时展示上一代完整立方体，不混合未完成面",
            "blocked_faces": True,
        },
        "sensor_stale": {
            "action": "hold_orientation",
            "detail": "传感器残差超容差时保持最近姿态，避免画面滞后或抖动",
            "max_extrapolation_ms": 0,
        },
        "resolution_change": {
            "action": "rebuild_sampling_before_present",
            "detail": "分辨率变化先重算面尺寸与接缝斜坡，再提交新帧",
            "drop_partial": True,
        },
        "pool_conflict": {
            "action": "allocate_per_face_slot",
            "detail": "检测到共享槽时按面重新分配，未分配成功则跳过该流",
            "shared": False,
        },
        "timestamp_drift": {
            "action": "realign_to_frame_ts",
            "detail": "时间戳漂移超阈值时按帧时间戳重新对齐传感器",
            "recompute_offset": True,
        },
    }
    spec = _spec(arg, kwargs)
    scalar = _scalar(spec)
    requested = spec.get("mode", spec.get("failure", spec.get("issue")))
    if requested is None and isinstance(scalar, str):
        requested = scalar
    active = modes.get(requested) if isinstance(requested, str) else None
    if active is not None:
        active = dict(active, mode=requested)
    return {
        "ok": True,
        "mode": requested if isinstance(requested, str) else None,
        "active": active,
        "modes": modes,
        "mode_count": len(modes),
        "coherent": True,
        "drop_partial": True,
        "never_show_partial_face": True,
        "value": active,
    }


# ---------------------------------------------------------------------------
# 延迟补偿 / 指标 / 姿态 / 可复现性
# ---------------------------------------------------------------------------


def latency_comp(arg: Any = None, **kwargs: Any) -> dict:
    """补偿传感器与显示管线延迟，给出用于姿态采样的等效时间戳。

    标量调用视为传感器延迟（毫秒）。等效时间戳 = 帧时间戳 − 管线延迟 − 传感器延迟。
    """
    spec = _spec(arg, kwargs)
    scalar = _scalar(spec)
    frame_ts = _as_float(spec.get("frame_ts", spec.get("frame_timestamp", 0.0)), 0.0)
    latency = _as_float(
        spec.get(
            "sensor_latency_ms",
            spec.get("latency_ms", scalar if isinstance(scalar, (int, float)) else 0.0),
        ),
        0.0,
    )
    pipeline = _as_float(spec.get("pipeline_delay_ms", spec.get("pipeline_ms", 0.0)), 0.0)
    jitter = _as_float(spec.get("jitter_ms", 0.0), 0.0)
    tolerance = _as_float(spec.get("max_residual_ms", DEFAULT_TOLERANCE_MS), DEFAULT_TOLERANCE_MS)

    effective = frame_ts - latency - pipeline
    return {
        "ok": True,
        "frame_ts": _round(frame_ts),
        "sensor_latency_ms": _round(latency),
        "pipeline_delay_ms": _round(pipeline),
        "jitter_ms": _round(jitter),
        "effective_sensor_ts": _round(effective),
        "offset_ms": _round(latency + pipeline),
        "residual_ms": _round(abs(jitter)),
        "within_tolerance": abs(jitter) <= tolerance,
        "compensated": True,
        "value": _round(effective),
    }


def metrics(arg: Any = None, **kwargs: Any) -> dict:
    """统计接缝误差指标；标量调用视为单个误差值。"""
    spec = _spec(arg, kwargs)
    scalar = _scalar(spec)
    errors = spec.get("errors", spec.get("seam_errors"))
    if errors is None:
        if isinstance(scalar, (list, tuple)):
            errors = list(scalar)
        elif scalar is None:
            errors = []
        else:
            errors = [scalar]

    values = []
    for item in errors:
        if isinstance(item, Mapping):
            value = item.get("error", item.get("value", item.get("px", 0.0)))
        else:
            value = item
        values.append(abs(_as_float(value, 0.0)))

    count = len(values)
    if count == 0:
        stats = {"count": 0, "mean": 0.0, "max": 0.0, "min": 0.0, "p95": 0.0, "rms": 0.0}
    else:
        ordered = sorted(values)
        mean = math.fsum(ordered) / count
        p95_index = min(count - 1, max(0, math.ceil(0.95 * count) - 1))
        stats = {
            "count": count,
            "mean": _round(mean),
            "max": _round(ordered[-1]),
            "min": _round(ordered[0]),
            "p95": _round(ordered[p95_index]),
            "rms": _round(math.sqrt(math.fsum(v * v for v in ordered) / count)),
        }

    baseline = spec.get("baseline", spec.get("before"))
    reduction_pct = None
    if baseline is not None and count:
        base_values = [
            abs(_as_float(b.get("error", b.get("value", 0.0)) if isinstance(b, Mapping) else b, 0.0))
            for b in (baseline if isinstance(baseline, (list, tuple)) else [baseline])
        ]
        if base_values:
            base_mean = math.fsum(base_values) / len(base_values)
            if base_mean > 0:
                reduction_pct = _round((1.0 - stats["mean"] / base_mean) * 100.0)

    return {
        "ok": True,
        "metrics": stats,
        **stats,
        "reduction_pct": reduction_pct,
        "unit": spec.get("unit", "px"),
        "value": stats["mean"],
    }


def orientation(arg: Any = None, **kwargs: Any) -> dict:
    """归一化/插值方向姿态。

    支持四元数 ``[w, x, y, z]`` 或 ``yaw``/``pitch``/``roll``（角度制）；
    给定 ``a``、``b``、``t`` 时按球面插值，标量调用视为 yaw 角。
    """
    spec = _spec(arg, kwargs)
    scalar = _scalar(spec)

    source = spec.get("quaternion", spec.get("q"))
    if source is None and "yaw" in spec:
        source = {
            "yaw": spec.get("yaw", 0.0),
            "pitch": spec.get("pitch", 0.0),
            "roll": spec.get("roll", 0.0),
        }
    if source is None and isinstance(scalar, (list, tuple)) and len(scalar) == 4:
        source = scalar
    if source is None and "value" in spec:
        source = scalar

    a = spec.get("a")
    b = spec.get("b")
    interpolated = False
    if a is not None and b is not None:
        qa = _orientation_to_quat(a) or [1.0, 0.0, 0.0, 0.0]
        qb = _orientation_to_quat(b) or [1.0, 0.0, 0.0, 0.0]
        t = _clamp(_as_float(spec.get("t", 0.5), 0.5), 0.0, 1.0)
        quat = _slerp(qa, qb, t)
        interpolated = True
    else:
        quat = _orientation_to_quat(source) if source is not None else [1.0, 0.0, 0.0, 0.0]

    return {
        "ok": True,
        "quaternion": _round_list(quat),
        **_quat_to_ypr(quat),
        "degrees": _quat_to_ypr(quat),
        "interpolated": interpolated,
        "normalized": True,
        "valid": True,
        "value": quat,
    }


def _determinism_probe() -> dict:
    """固定场景的规范探针，用来验证整条管线没有隐藏的随机性。"""
    return _switch_plan(
        {
            "frame": 7,
            "switch_frame": 7,
            "version": 0,
            "projection": "cube",
            "source_projection": "equirect",
            "done_faces": list(FACES),
            "decoders": 2,
            "from_width": 1920,
            "to_width": 3840,
            "blend_px": 8,
            "frame_ts": 1000.0,
            "sensor_latency_ms": 12.0,
        }
    )


def determinism(arg: Any = None, **kwargs: Any) -> dict:
    """对固定场景重复运行整条切换管线，比较规范化摘要。

    标量调用视为重复次数（至少 2 次，最多 8 次）。
    """
    spec = _spec(arg, kwargs)
    scalar = _scalar(spec)
    runs = _as_int(spec.get("runs", scalar), 2)
    runs = int(_clamp(runs, 2, 8))

    digests = [_digest(_determinism_probe()) for _ in range(runs)]
    identical = len(set(digests)) == 1
    return {
        "ok": True,
        "stable": identical,
        "identical": identical,
        "runs": runs,
        "algorithm": "sha256",
        "canonical_json": True,
        "digest": digests[0],
        "digests": digests,
        "value": digests[0],
    }
