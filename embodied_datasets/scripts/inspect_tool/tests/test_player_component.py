import json


def _minimal_payload():
    return {
        "fps": 10.0,
        "views": ["observation.image"],
        "videos": {
            "observation.image": {
                "raw": {
                    "url": "http://127.0.0.1:9000/clip.mp4", "from_timestamp": 0.0, "to_timestamp": 0.5,
                    "element_id": "video-observation.image-raw",
                },
                "final": None,
            }
        },
        "raw_frame_count": 5,
        "final_frame_count": 0,
        "charts": {
            "state": [
                {
                    "label": "arm1_joint", "populated": True, "figure": {"data": [], "layout": {}},
                    "div_id": "chart-state-arm1_joint",
                },
            ],
            "action": [],
        },
    }


def test_build_html_embeds_payload_json():
    from player_component import build_html

    payload = _minimal_payload()
    html = build_html(payload)
    assert json.dumps(payload) in html


def test_build_html_embeds_plotly_library_inline():
    from player_component import _PLOTLY_JS, build_html

    html = build_html(_minimal_payload())
    assert _PLOTLY_JS in html


def test_build_html_has_no_leftover_placeholders():
    from player_component import build_html

    html = build_html(_minimal_payload())
    assert "__PLOTLY_JS__" not in html
    assert "__PAYLOAD_JSON__" not in html


def test_build_html_references_every_chart_div_id():
    from player_component import build_html

    html = build_html(_minimal_payload())
    assert 'id="chart-state-arm1_joint"' in html


def test_build_html_shows_placeholder_for_missing_final_video():
    from player_component import build_html

    html = build_html(_minimal_payload())
    assert "final not available" in html
    assert 'id="video-observation.image-final"' not in html


def test_build_html_shows_placeholder_for_missing_raw_video():
    from player_component import build_html

    payload = _minimal_payload()
    payload["videos"]["observation.image"] = {
        "raw": None,
        "final": {
            "url": "http://127.0.0.1:9000/final.mp4", "from_timestamp": 0.0, "to_timestamp": 0.5,
            "element_id": "video-observation.image-final",
        },
    }
    html = build_html(payload)
    assert "raw not available" in html
    assert 'id="video-observation.image-raw"' not in html


def test_build_html_renders_multiple_views():
    from player_component import build_html

    payload = _minimal_payload()
    payload["views"] = ["observation.image", "observation.wrist_image"]
    payload["videos"]["observation.wrist_image"] = {
        "raw": {
            "url": "http://127.0.0.1:9000/wrist_raw.mp4", "from_timestamp": 0.0, "to_timestamp": 0.5,
            "element_id": "video-observation.wrist_image-raw",
        },
        "final": None,
    }
    html = build_html(payload)
    assert 'id="video-observation.image-raw"' in html
    assert "final not available" in html
    assert 'id="video-observation.wrist_image-raw"' in html
    assert html.count("final not available") == 2


def test_build_html_renders_multiple_chart_bands_including_populated_action():
    from player_component import build_html

    payload = _minimal_payload()
    payload["charts"]["action"] = [
        {
            "label": "gripper", "populated": True, "figure": {"data": [], "layout": {}},
            "div_id": "chart-action-gripper",
        },
    ]
    html = build_html(payload)
    assert 'id="chart-state-arm1_joint"' in html
    assert 'id="chart-action-gripper"' in html


def test_build_html_marks_unpopulated_band_label_but_not_populated_band():
    from player_component import build_html

    payload = _minimal_payload()
    payload["charts"]["state"].append(
        {
            "label": "arm2_joint", "populated": False, "figure": {"data": [], "layout": {}},
            "div_id": "chart-state-arm2_joint",
        }
    )
    html = build_html(payload)
    assert "arm2_joint (unpopulated)" in html
    assert "arm1_joint (unpopulated)" not in html
