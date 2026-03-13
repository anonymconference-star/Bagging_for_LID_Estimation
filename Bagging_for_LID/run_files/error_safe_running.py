from Bagging_for_LID.RunningEstimators.Running2 import *

def setup_logging(log_file: str = "run.log") -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),  # console
        ],
    )

def run_task_safely(func, *args, **kwargs):
    name = getattr(func, "__name__", str(func))
    try:
        logging.info("Starting %s", name)
        result = func(*args, **kwargs)
        logging.info("Finished %s", name)
        return True, result
    except Exception as e:
        print(f"[ERROR] {name} failed: {e}")
        logging.exception("Task %s failed with an exception", name)
        return False, None

def consume_and_plot(task_key, result_dict, plot_tasks):
    for plot_fn, plot_kwargs in plot_tasks.get(task_key, []):
        plotting_across_results_dict(result_dict, plot_fn, **plot_kwargs)
        plt.close('all')