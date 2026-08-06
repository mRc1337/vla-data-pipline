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
