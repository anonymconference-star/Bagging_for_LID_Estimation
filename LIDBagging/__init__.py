from importlib import import_module
import sys

# Import your actual new package
new = import_module("Bagging_for_LID")

# Make LIDBagging refer to Bagging_for_LID
sys.modules["LIDBagging"] = new

# Re-export everything (submodules will be resolved correctly)
globals().update({
    name: getattr(new, name)
    for name in dir(new)
    if not name.startswith("_")
})