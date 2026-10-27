"""Model checking for the shielding code with stormpy >= 2.0.

stormpy 2.0 checks with a sound environment by default (`stormpy.environments.default`). The shields were built with the
values of stormpy 1.x's default, non-sound value iteration (topological, precision 1e-6, relative termination), which a
plain `stormpy.Environment()` still gives: vmin and vmax of bridge_crossing come out bit-identical to stormpy 1.14, in
2 s, while the sound default had not finished after 5 minutes on that 400-state model (its risk builds up very slowly).
"""
import stormpy


def classic_environment():
    """The environment of stormpy 1.x's default model checking: non-sound value iteration, precision 1e-6 relative."""
    return stormpy.Environment()


def model_check(model, formula):
    """stormpy.model_checking with the classic environment; `formula` is a property or a string in the PRISM syntax."""
    if isinstance(formula, str):
        formula = stormpy.parse_properties(formula)[0]
    return stormpy.model_checking(model, formula, environment=classic_environment())
