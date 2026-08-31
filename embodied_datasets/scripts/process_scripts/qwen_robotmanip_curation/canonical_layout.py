"""Canonical 128-D layout from ``公开数据集格式规范.docx``.

Action index 68 is the unused padding member of the document's 35-D ARM2
slice [34:69].  The actual ARM2 payload mirrors ARM1 and occupies [34:68].
It must remain zero with mask=False.
"""

STATE_DIM = 128
STATE_JOINT = (0, 7)
STATE_EEF = (7, 14)
STATE_GRIPPER = (14, 35)
STATE_ARM2 = (35, 70)
STATE_RESERVE = (70, 128)

ACTION_DIM = 128
ACTION_JOINT = (0, 7)
ACTION_EEF_POS = (7, 10)
ACTION_EEF_ROT = (10, 13)
ACTION_GRIPPER = (13, 34)
ACTION_ARM2 = (34, 69)
ACTION_ARM2_PAYLOAD = (34, 68)
ACTION_ARM2_PADDING = 68
ACTION_RESERVE = (69, 128)

STATE_JOINT_INDICES = frozenset(range(0, 7)) | frozenset(range(35, 42))
ACTION_JOINT_INDICES = frozenset(range(0, 7)) | frozenset(range(34, 41))

LAYOUT = {
    "source": "/home/pai/zxw/公开数据集格式规范.docx",
    "state": {
        "dimension": STATE_DIM,
        "joint": list(STATE_JOINT),
        "eef_position_quaternion": list(STATE_EEF),
        "gripper_or_hand": list(STATE_GRIPPER),
        "arm2": list(STATE_ARM2),
        "reserve": list(STATE_RESERVE),
    },
    "action": {
        "dimension": ACTION_DIM,
        "joint_delta": list(ACTION_JOINT),
        "eef_position_delta": list(ACTION_EEF_POS),
        "eef_rotation_delta_axis_angle": list(ACTION_EEF_ROT),
        "gripper_or_hand": list(ACTION_GRIPPER),
        "arm2": list(ACTION_ARM2),
        "arm2_payload": list(ACTION_ARM2_PAYLOAD),
        "arm2_padding_zero_mask_false": ACTION_ARM2_PADDING,
        "reserve": list(ACTION_RESERVE),
    },
}
