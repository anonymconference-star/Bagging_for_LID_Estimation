"""
Lightweight version of LID_experiment that holds only the attributes
needed by the plotting functions (MSEbars.py, VariableInteraction.py,
SpiderCharts.py) and their preprocessing helpers
(optimize_across_parameter_results.py).

No heavy imports (datasets, estimators, skdim, etc.).
"""


class LID_experiment_light:
    """Drop-in replacement for LID_experiment for plotting purposes only."""

    __slots__ = (
        "n", "k", "sr", "Nbag",
        "dataset_name", "lid", "dim",
        "pre_smooth", "post_smooth", "t",
        "estimator_name", "bagging_method",
        "submethod_0", "submethod_error",
        "params", "string",
        "total_mse", "total_bias2", "total_var",
        "log_total_mse", "log_total_bias2", "log_total_var",
    )

    # The attribute names that are copied from a full LID_experiment
    _COPY_ATTRS = __slots__

    def __init__(self):
        for attr in self.__slots__:
            setattr(self, attr, None)

    @classmethod
    def from_experiment(cls, exp):
        """Create a light copy from a full LID_experiment."""
        light = cls()
        for attr in cls._COPY_ATTRS:
            setattr(light, attr, getattr(exp, attr, None))
        return light

    @classmethod
    def from_experiments(cls, experiments):
        """Convert a list of full LID_experiment objects into a list of light copies."""
        return [cls.from_experiment(exp) for exp in experiments]
