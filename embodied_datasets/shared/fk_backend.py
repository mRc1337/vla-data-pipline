"""ikpy-based forward-kinematics wrapper for Stage4 FK consistency
checking. See
docs/superpowers/specs/2026-07-17-process-scripts-cleaning-alignment-design.md
section 4 for why ikpy (pure-Python, pip-installable) was chosen over
Pinocchio.

Note on ikpy's link/joint mapping (verified against the installed ikpy
version): ikpy represents each URDF *joint* as one of its own "links" --
URDF links themselves carry no transform and are discarded. So a chain
built from N URDF links joined by revolute/fixed joints has one ikpy
link per joint, including a synthetic fixed "Base link" at index 0.
`joint_positions` passed to `forward()` should contain exactly one angle
per *active* (non-fixed) joint, in URDF joint order.
"""
from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
from ikpy.chain import Chain
from scipy.spatial.transform import Rotation

# ikpy's default active_links_mask marks every link (including fixed
# joints) as active, then warns that fixed links are inert anyway. That
# warning is expected here -- we always pass a full angle vector and
# derive which entries matter from joint_type ourselves -- so it is
# suppressed narrowly (only around the parsing call, only this specific
# UserWarning) rather than left to leak into every caller's test output.
_IKPY_FIXED_LINK_ACTIVE_WARNING = r".*is of type 'fixed' but set as active"


class FkChain:
    def __init__(self, urdf_path: str):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=_IKPY_FIXED_LINK_ACTIVE_WARNING, category=UserWarning
            )
            self._chain = Chain.from_urdf_file(urdf_path)
        self._active_link_indices = [
            i for i, link in enumerate(self._chain.links) if link.joint_type != "fixed"
        ]

    def forward(self, joint_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        full_angles = np.zeros(len(self._chain.links))
        for idx, angle in zip(self._active_link_indices, joint_positions):
            full_angles[idx] = angle
        transform = self._chain.forward_kinematics(full_angles)
        position = np.asarray(transform[:3, 3], dtype=float)
        quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
        return position, quaternion
