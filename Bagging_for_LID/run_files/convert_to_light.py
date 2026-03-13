"""
Load full LID_experiment pkl files and re-save them as lightweight
LID_experiment_light objects (keeping only the attributes needed for plotting).

Saved files get a 'light_' prefix so the originals are untouched.
"""
import os
import sys
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
p = str(PROJECT_ROOT)
if p not in sys.path:
    sys.path.insert(0, p)

from Bagging_for_LID.experiment_class_light import LID_experiment_light

# ── configuration ──────────────────────────────────────────────────────────
pkl_directory = r"C:\pkls"
save_name = "mergedresult"           # prefix used when the experiments were saved
light_prefix = "light_"             # new prefix prepended to each pkl name

# These are the sub-keys produced by general_result_generator via
# param_dicts_general.  Each one maps to a pkl file named
# "{save_name}_{task}_{estimator}_{test_type}".
task_keys = {
    "Bagging_and_smoothing_test":                          ("effectiveness_test_base_param_dict", ["mle", "mada", "tle"], ["smooth"]),
    "Number_of_bags_test":                                 ("Nbag_test_base_param_dict",          ["mle", "mada", "tle"], ["variable"]),
    "Sampling_rate_test":                                  ("sr_prog_test_base_param_dict",        ["mle", "mada", "tle"], ["variable"]),
    "Interaction_of_sampling_rate_and_number_of_bags_test": ("interaction_sr_Nbag_test_base_param_dict", ["mle", "mada", "tle"], ["variable"]),
    "Interaction_of_k_and_sampling_rate_test":              ("interaction_sr_k_test_base_param_dict",    ["mle", "mada", "tle"], ["variable"]),
}

def build_pkl_names():
    """Return the list of pkl file names (without directory) that were saved."""
    names = []
    for task_name, (_, estimators, test_types) in task_keys.items():
        for est in estimators:
            for tt in test_types:
                names.append(f"{save_name}_{task_name}_{est}_{tt}")
    return names


def load_pkl(directory, name):
    filepath = os.path.join(directory, name)
    with open(filepath, "rb") as fh:
        return pickle.load(fh)


def save_pkl(data, directory, name):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, name)
    with open(filepath, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    pkl_names = build_pkl_names()
    converted = 0
    skipped = 0

    for name in pkl_names:
        src = os.path.join(pkl_directory, name)
        if not os.path.exists(src):
            print(f"[SKIP] {name}  (file not found)")
            skipped += 1
            continue

        print(f"[LOAD] {name}  ...", end="", flush=True)
        experiments = load_pkl(pkl_directory, name)
        light_experiments = LID_experiment_light.from_experiments(experiments)

        out_name = f"{light_prefix}{name}"
        save_pkl(light_experiments, pkl_directory, out_name)

        src_size = os.path.getsize(src) / (1024 * 1024)
        dst_size = os.path.getsize(os.path.join(pkl_directory, out_name)) / (1024 * 1024)
        print(f"  {len(experiments)} exps  {src_size:.1f} MB -> {dst_size:.1f} MB  [SAVED] {out_name}")
        converted += 1

        # free memory before next file
        del experiments, light_experiments

    print(f"\nDone. Converted: {converted}, Skipped: {skipped}")
