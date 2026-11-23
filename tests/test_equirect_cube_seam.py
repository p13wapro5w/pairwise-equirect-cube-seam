import pytest
import media.equirect_cube_seam as mod

CASES = [
    ("mapping_version", (1)),
    ("sensor_align", (1)),
    ("face_done", (1)),
    ("pool_per_face", (1)),
    ("resolution_recalc", (1)),
    ("seam_blend", (1)),
    ("projection_switch", (1)),
    ("validation", (1)),
    ("audit", (1)),
    ("fallback", (1)),
    ("latency_comp", (1)),
    ("metrics", (1)),
    ("orientation", (1)),
    ("face_order", (1)),
    ("determinism", (1)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
