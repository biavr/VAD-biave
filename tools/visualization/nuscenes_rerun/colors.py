"""The nuScenes-lidarseg 32-class label table: id -> (class name, RGB color).

The names and colors come from the official nuscenes-devkit color palette
(``nuscenes.utils.color_map.get_colormap``). The devkit keys that palette by
class *name*; here it is re-indexed by the integer class *id* used inside
``*_lidarseg.bin`` files, using the standard nuScenes-lidarseg ordering
(0 = noise, ..., 31 = vehicle.ego) established by the challenge's own
category table.
"""

from __future__ import annotations

LIDARSEG_CLASSES: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("noise", (0, 0, 0)),
    ("animal", (70, 130, 180)),
    ("human.pedestrian.adult", (0, 0, 230)),
    ("human.pedestrian.child", (135, 206, 235)),
    ("human.pedestrian.construction_worker", (100, 149, 237)),
    ("human.pedestrian.personal_mobility", (219, 112, 147)),
    ("human.pedestrian.police_officer", (0, 0, 128)),
    ("human.pedestrian.stroller", (240, 128, 128)),
    ("human.pedestrian.wheelchair", (138, 43, 226)),
    ("movable_object.barrier", (112, 128, 144)),
    ("movable_object.debris", (210, 105, 30)),
    ("movable_object.pushable_pullable", (105, 105, 105)),
    ("movable_object.trafficcone", (47, 79, 79)),
    ("static_object.bicycle_rack", (188, 143, 143)),
    ("vehicle.bicycle", (220, 20, 60)),
    ("vehicle.bus.bendy", (255, 127, 80)),
    ("vehicle.bus.rigid", (255, 69, 0)),
    ("vehicle.car", (255, 158, 0)),
    ("vehicle.construction", (233, 150, 70)),
    ("vehicle.emergency.ambulance", (255, 83, 0)),
    ("vehicle.emergency.police", (255, 215, 0)),
    ("vehicle.motorcycle", (255, 61, 99)),
    ("vehicle.trailer", (255, 140, 0)),
    ("vehicle.truck", (255, 99, 71)),
    ("flat.driveable_surface", (0, 207, 191)),
    ("flat.other", (175, 0, 75)),
    ("flat.sidewalk", (75, 0, 75)),
    ("flat.terrain", (112, 180, 60)),
    ("static.manmade", (222, 184, 135)),
    ("static.other", (255, 228, 196)),
    ("static.vegetation", (0, 175, 0)),
    ("vehicle.ego", (255, 240, 245)),
)


def annotation_context() -> list[tuple[int, str, tuple[int, int, int]]]:
    """Build the ``(id, label, color)`` triples expected by `rr.AnnotationContext`."""
    return [(class_id, name, color) for class_id, (name, color) in enumerate(LIDARSEG_CLASSES)]
