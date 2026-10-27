"""Loads a benchmark (`<project>/sketch.templ` + `sketch.props`) as an explicit stormpy model, for stormpy >= 2.0 and
paynt >= 0.5.0.

UMB exports (Storm's binary explicit-model format, a gzip archive) are read with stormpy alone. PRISM sketches go through
the installed paynt package: its loader rewrites `observables` declarations, adds missing choice labels, turns state
rewards into state-action rewards and makes POMDPs canonic, and `PomdpColoredMdpFactory.pomdp` is that model, the same
one the old `PomdpQuotient.pomdp` held.

The repository root contains an old fork of paynt in `paynt/`, which shadows the installed package whenever the root is
on sys.path (e.g. for every script run from there). The fork needs paynt's old Python API and is not compatible with
paynt 0.5.0 / stormpy 2.0, so the installed package is imported with the repository root taken off sys.path.
"""
import importlib
import os
import sys

import stormpy


class LoadedSketch:
    """What the shielding code needs from a loaded sketch: the explicit model, read as `.pomdp`."""

    def __init__(self, pomdp):
        self.pomdp = pomdp


def _is_umb_archive(sketch_path):
    # UMB exports are gzip-compressed archives, PRISM sketches are plain text; both are called sketch.templ.
    with open(sketch_path, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def installed_paynt():
    """The installed paynt package, never the fork in the repository root."""
    loaded = sys.modules.get("paynt")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    fork = os.path.join(repo_root, "paynt")
    if loaded is not None:
        if os.path.dirname(os.path.abspath(loaded.__file__)) == fork:
            raise ImportError(f"the paynt fork in {fork} is already imported; it is not compatible with paynt >= 0.5.0")
        return loaded
    saved = sys.path[:]
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != repo_root]
    try:
        return importlib.import_module("paynt")
    finally:
        sys.path[:] = saved


def load_sketch(project_path):
    project_path = os.path.abspath(project_path)
    sketch_path = os.path.join(project_path, "sketch.templ")
    properties_path = os.path.join(project_path, "sketch.props")
    if _is_umb_archive(sketch_path):
        return LoadedSketch(stormpy.developer.storage.sparse_model_from_umb(stormpy.io.import_umb(sketch_path)))
    installed_paynt()
    import paynt.parser.sketch  # resolves to the installed package now that it is in sys.modules
    factory, _ = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)
    if not hasattr(factory, "pomdp"):  # the old loader gave an MdpQuotient without .pomdp here, too
        raise ValueError(f"{sketch_path} is not a POMDP sketch; PRISM sketches have to be POMDPs (MDPs with full "
                         "observability included), UMB exports can be either")
    return LoadedSketch(factory.pomdp)
