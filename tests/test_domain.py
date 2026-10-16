import pytest

from vlogflow.domain import validate_scenes

MEDIA = [{"id": "m1", "duration": 5.0}]


def scene(**kw):
    return dict(
        id="s1", media_id="m1", start=0, end=5, description="street", narration="Hello.", ambient=0.1, **kw
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"media_id": "missing"},
        {"start": -1},
        {"end": 6},
        {"start": 3, "end": 2},
        {"end": float("nan")},
        {"end": float("inf")},
        {"narration": " "},
        {"ambient": 1.1},
    ],
)
def test_rejects_unrenderable_scenes(changes):
    s = scene()
    s.update(changes)
    with pytest.raises(ValueError):
        validate_scenes(MEDIA, [s])


def test_duplicate_scene_ids_cannot_alias_outputs():
    with pytest.raises(ValueError):
        validate_scenes(MEDIA, [scene(), scene()])


def test_returns_normalized_valid_scene():
    assert validate_scenes(MEDIA, [scene()])[0]["narration"] == "Hello."
