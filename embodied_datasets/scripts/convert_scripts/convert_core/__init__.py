"""Format-agnostic infrastructure shared by every convert_scripts reader.

Readers (see ``readers/``) turn one raw dataset format into the plan/frame
contract defined in :mod:`convert_core.episode_spec`; everything in this
package operates only on that contract and never on a specific source format.
"""
