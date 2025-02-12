import logging
import os
import re
import subprocess
import sys
import threading
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import optuna
import yaml
from docx import Document
from fuzzywuzzy import fuzz


def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a logger with a standardized format.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_logger(__name__)


class ConfigManager:
    """
    Handles loading and saving of a YAML configuration file.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> dict:
        """
        Loads and returns the configuration from a YAML file.
        """
        if not os.path.exists(self.config_path):
            logger.error(f"Configuration file not found: {self.config_path}")
            sys.exit(1)
        with open(self.config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def save_config(self, config: dict) -> None:
        """
        Saves the given configuration (or the current configuration if None) to the YAML file.
        """
        if not config:
            config = self.config
        with open(self.config_path, "w", encoding="utf-8") as file:
            yaml.dump(config, file, default_flow_style=False, allow_unicode=True)


class ChordExtractor:
    """
    Extracts chord strings from a DOCX file using a regular expression.
    """

    CHORD_PATTERN = re.compile(
        r"\b[A-Ga-g][#b]?(?:m|maj|sus|dim|aug|add)?\d*(?:/[A-Ga-g][#b]?)?\b"
    )

    @classmethod
    def extract_chords_from_docx(cls, file_path: str) -> list:
        """
        Returns a list of chords extracted from the DOCX file at the given path.
        """
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return []
        doc = Document(file_path)
        chords = []
        for para in doc.paragraphs:
            matches = cls.CHORD_PATTERN.findall(para.text)
            chords.extend(matches)
        return chords


class ScoreEvaluator:
    """
    Computes a score by comparing generated chords with a reference.
    """

    @staticmethod
    def evaluate(reference: list, generated: list) -> float:
        """
        Returns a score computed as:
            score = (3 * number of correct matches) - (2 * false positives) - (2 * false negatives)
        A correct match is counted if the fuzzy similarity exceeds 80.
        Returns -100 if either list is empty.
        """
        if not reference or not generated:
            return -100.0
        correct_matches = sum(
            fuzz.ratio(r, g) > 75 for r, g in zip(reference, generated)
        )
        false_positives = max(0, len(generated) - len(reference))
        false_negatives = max(0, len(reference) - len(generated))
        score = (3 * correct_matches) - (false_positives) - (false_negatives)
        logger.info(
            f"Score Breakdown - Correct: {correct_matches}, False Positives: {false_positives}, False Negatives: {false_negatives}"
        )
        logger.info(f"Final Score: {score:.2f}")
        return score


class OptimizationRunner:
    """
    Encapsulates the optimization process using Optuna.
    Updates the configuration, runs the main process, and computes the objective score.
    """

    def __init__(
        self, config_manager: ConfigManager, results_folder: str, reference_file: str
    ):
        self.config_manager = config_manager
        self.results_folder = results_folder
        self.reference_file = reference_file

    def run_main_script(self) -> bool:
        """
        Executes the main process (main.py) as a subprocess.
        Returns True if successful, False otherwise.
        """
        process = subprocess.run(
            ["python", "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if process.returncode != 0:
            logger.error("main.py execution failed.")
            logger.error(f"Error Output: {process.stderr.strip()}")
            return False
        return True

    def get_latest_output_file(self) -> str:
        """
        Returns the path to the most recently modified file in the results folder.
        Returns an empty string if no file is found.
        """
        try:
            files = sorted(
                os.listdir(self.results_folder),
                key=lambda f: os.path.getmtime(os.path.join(self.results_folder, f)),
            )
        except Exception as e:
            logger.error(f"Error listing files in {self.results_folder}: {e}")
            return ""
        if not files:
            logger.error("No output files generated.")
            return ""
        return os.path.join(self.results_folder, files[-1])

    def objective(self, trial: optuna.Trial) -> float:
        """
        The objective function for the Optuna study.
        Randomly suggests values for hyperparameters, updates the configuration,
        runs the main process, and returns the computed score.
        """
        sensitivity = trial.suggest_float("sensitivity", 0.0, 1.0)
        onset_delta = trial.suggest_float("onset_delta", 0.0, 1.0)
        hop_length = trial.suggest_categorical("hop_length", list(range(64, 513, 64)))
        sensitivity = round(sensitivity, 2)
        onset_delta = round(onset_delta, 2)
        config = self.config_manager.config
        config["chord_detection"]["sensitivity"] = sensitivity
        config["chord_detection"]["onset_delta"] = onset_delta
        config["chord_detection"]["hop_length"] = hop_length
        self.config_manager.save_config(config)
        logger.info(f"Trial {trial.number} Parameters: {config['chord_detection']}")
        if not self.run_main_script():
            return -100.0
        output_file = self.get_latest_output_file()
        if not output_file:
            return -100.0
        generated_chords = ChordExtractor.extract_chords_from_docx(output_file)
        reference_chords = ChordExtractor.extract_chords_from_docx(self.reference_file)
        return ScoreEvaluator.evaluate(reference_chords, generated_chords)


def create_plots():
    """
    Creates and returns a tuple (fig, axes) containing the matplotlib figure and a dictionary of axes.
    """
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)
    axes = {
        "score_evolution": fig.add_subplot(gs[0, 0]),
        "param_importance": fig.add_subplot(gs[0, 1]),
        "parallel_coords": fig.add_subplot(gs[0, 2]),
    }
    gs_slice = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 0])
    axes["slice1"] = fig.add_subplot(gs_slice[0])
    axes["slice2"] = fig.add_subplot(gs_slice[1])
    axes["slice3"] = fig.add_subplot(gs_slice[2])
    axes["contour"] = fig.add_subplot(gs[1, 1])
    axes["heatmap"] = fig.add_subplot(gs[1, 2])
    return fig, axes


# Global colorbar variables for contour and heatmap
contour_colorbar = None
heatmap_colorbar = None

# Global flag to indicate if the figure is closing
is_closing = False


def update_score_evolution(ax, trial_numbers, scores):
    """
    Updates the score evolution plot.

    Clears the given axes and plots the trial numbers against the scores.

    Parameters:
        ax (matplotlib.axes.Axes): The axes to update.
        trial_numbers (list): List of trial numbers.
        scores (list): List of score values corresponding to the trials.
    """
    ax.clear()
    ax.plot(trial_numbers, scores, marker="o")
    ax.set_title("Score Evolution")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Score")


def update_param_importance(ax, study):
    """
    Updates the parameter importance plot.

    Clears the given axes and, if enough trial data exists, plots the parameter importance
    computed by Optuna. If there is insufficient data or an error occurs, an appropriate
    message is displayed.

    Parameters:
        ax (matplotlib.axes.Axes): The axes to update.
        study (optuna.study.Study): The Optuna study containing trial data.
    """
    ax.clear()
    if len(study.trials) <= 1:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
        return
    try:
        import optuna.importance as oi

        if imp := oi.get_param_importances(study):
            ax.bar(list(imp.keys()), list(imp.values()), color="orange")
            ax.set_title("Parameter Importance")
            ax.set_ylabel("Importance")
        else:
            ax.text(0.5, 0.5, "No importance computed", ha="center", va="center")
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")


def update_parallel_coords(ax, trial_numbers, sv, od, hl):
    """
    Updates the parallel coordinates plot.

    Clears the given axes and plots the normalized hyperparameter values for each trial.
    The normalization for sensitivity and onset_delta is identity (since their range is [0,1]),
    and for hop_length it is normalized to the range [0,1] from [64,512].

    Parameters:
        ax (matplotlib.axes.Axes): The axes to update.
        trial_numbers (list): List of trial numbers.
        sv (list): List of sensitivity values.
        od (list): List of onset_delta values.
        hl (list): List of hop_length values.
    """
    ax.clear()
    for i in range(len(trial_numbers)):
        norm_s = sv[i]
        norm_o = od[i]
        norm_h = (hl[i] - 64) / (512 - 64)
        ax.plot([0, 1, 2], [norm_s, norm_o, norm_h], marker="o", alpha=0.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Sensitivity", "Onset Delta", "Hop Length"])
    ax.set_title("Parallel Coordinates")


def update_slice_plots(axes, sv, od, hl, scores):
    """
    Updates the three slice plots that show the relationship between each hyperparameter and the score.

    Each slice plot is cleared and updated with a scatter plot:
      - "slice1" plots Sensitivity vs. Score (red markers).
      - "slice2" plots Onset Delta vs. Score (blue markers).
      - "slice3" plots Hop Length vs. Score (green markers).

    Parameters:
        axes (dict): Dictionary containing the axes for "slice1", "slice2", and "slice3".
        sv (list): List of sensitivity values.
        od (list): List of onset_delta values.
        hl (list): List of hop_length values.
        scores (list): List of score values.
    """
    axes["slice1"].clear()
    axes["slice1"].scatter(sv, scores, color="red")
    axes["slice1"].set_title("Sensitivity vs Score")
    axes["slice1"].set_xlabel("Sensitivity")
    axes["slice1"].set_ylabel("Score")

    axes["slice2"].clear()
    axes["slice2"].scatter(od, scores, color="blue")
    axes["slice2"].set_title("Onset Delta vs Score")
    axes["slice2"].set_xlabel("Onset Delta")
    axes["slice2"].set_ylabel("Score")

    axes["slice3"].clear()
    axes["slice3"].scatter(hl, scores, color="green")
    axes["slice3"].set_title("Hop Length vs Score")
    axes["slice3"].set_xlabel("Hop Length")
    axes["slice3"].set_ylabel("Score")


def update_contour_plot(ax, sv, od, scores, contour_cb):
    """
    Updates the contour plot based on sensitivity and onset_delta.

    If there are not enough data points (<= 5), a message is displayed.
    Otherwise, a contour plot is drawn using tricontourf and the existing colorbar is updated.

    Parameters:
        ax (matplotlib.axes.Axes): The axes to update.
        sv (list): List of sensitivity values.
        od (list): List of onset_delta values.
        scores (list): List of score values.
        contour_cb: The existing contour colorbar object (or None).

    Returns:
        The updated contour colorbar object.
    """
    ax.clear()
    if len(sv) <= 5:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
        return contour_cb
    x = np.array(sv)
    y = np.array(od)
    z = np.array(scores)
    try:
        cont = ax.tricontourf(x, y, z, levels=14, cmap="viridis")
        if contour_cb is None:
            contour_cb = plt.gcf().colorbar(cont, ax=ax)
        else:
            contour_cb.update_normal(cont)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
    ax.set_title("Contour Plot (Sensitivity, Onset Delta)")
    ax.set_xlabel("Sensitivity")
    ax.set_ylabel("Onset Delta")
    return contour_cb


def update_heatmap(ax, sv, od, scores, heatmap_cb):
    """
    Updates the heatmap of hyperparameters vs. score.

    The heatmap is drawn as a scatter plot with the "coolwarm" colormap, where red indicates higher scores
    and blue indicates lower scores. If a heatmap colorbar already exists, it is updated.

    Parameters:
        ax (matplotlib.axes.Axes): The axes to update.
        sv (list): List of sensitivity values.
        od (list): List of onset_delta values.
        scores (list): List of score values.
        heatmap_cb: The existing heatmap colorbar object (or None).

    Returns:
        The updated heatmap colorbar object.
    """
    ax.clear()
    sc_plot = ax.scatter(sv, od, c=scores, cmap="coolwarm", s=100)
    ax.set_title("Hyperparameter Heatmap")
    ax.set_xlabel("Sensitivity")
    ax.set_ylabel("Onset Delta")
    if heatmap_cb is None:
        heatmap_cb = plt.gcf().colorbar(
            sc_plot, ax=ax, label="Score (red=high, blue=low)"
        )
    else:
        heatmap_cb.update_normal(sc_plot)
    return heatmap_cb


def update_all_plots(
    axes: dict,
    study: optuna.study.Study,
    trial: optuna.trial.FrozenTrial,
    contour_cb,
    hyper_data: dict,
):
    """
    Updates all plots with the latest trial data.

    The hyper_data dictionary must contain the following keys:
      'trial_numbers', 'scores', 'sensitivity_values', 'onset_delta_values', 'hop_length_values', 'times'.

    This function sequentially updates:
      - The Score Evolution plot.
      - The Parameter Importance plot.
      - The Parallel Coordinates plot.
      - The three slice plots.
      - The Contour plot.
      - The Heatmap of hyperparameters vs. score.

    Returns:
        The updated contour colorbar object.
    """
    tn = hyper_data["trial_numbers"]
    sc = hyper_data["scores"]
    sv = hyper_data["sensitivity_values"]
    od = hyper_data["onset_delta_values"]
    hl = hyper_data["hop_length_values"]

    update_score_evolution(axes["score_evolution"], tn, sc)
    update_param_importance(axes["param_importance"], study)
    update_parallel_coords(axes["parallel_coords"], tn, sv, od, hl)
    update_slice_plots(axes, sv, od, hl, sc)
    contour_cb = update_contour_plot(axes["contour"], sv, od, sc, contour_cb)
    global heatmap_colorbar
    heatmap_colorbar = update_heatmap(axes["heatmap"], sv, od, sc, heatmap_colorbar)
    plt.gcf().canvas.draw_idle()
    
    return contour_cb


def run_optimization(study, optimization_runner, n_trials, callback_fn):
    """
    Runs the optimization process using the provided callback.
    """
    study.optimize(
        optimization_runner.objective, n_trials=n_trials, callbacks=[callback_fn]
    )


def main():
    """
    Main function to run the hyperparameter optimization and update the plots.
    """
    global start_time, is_closing

    config_path = "config.yml"
    results_folder = "data/outputs/"
    reference_file = "data/reference/Braço Forte.docx"

    config_manager = ConfigManager(config_path)
    optimization_runner = OptimizationRunner(
        config_manager, results_folder, reference_file
    )
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler()
    )

    plt.ion()
    fig, axes = create_plots()
    try:
        fig.canvas.manager.window.wm_attributes("-topmost", 0)  # type: ignore
    except Exception as e:
        logger.warning(f"Could not set window attributes: {e}")

    def on_close(event):
        global is_closing
        is_closing = True

    fig.canvas.mpl_connect("close_event", on_close)
    plt.tight_layout()

    n_trials = 200
    start_time = time.time()

    hyper_data = {
        "trial_numbers": [],
        "scores": [],
        "sensitivity_values": [],
        "onset_delta_values": [],
        "hop_length_values": [],
        "times": [],
    }
    contour_cb = None

    def callback(study, trial):
        hyper_data["trial_numbers"].append(trial.number)
        hyper_data["scores"].append(trial.value)
        hyper_data["sensitivity_values"].append(trial.params.get("sensitivity", 0.95))
        hyper_data["onset_delta_values"].append(trial.params.get("onset_delta", 0.17))
        hyper_data["hop_length_values"].append(trial.params.get("hop_length", 512))
        hyper_data["times"].append(time.time() - start_time if start_time else 0.0)
        nonlocal contour_cb
        contour_cb = update_all_plots(axes, study, trial, contour_cb, hyper_data)

    opt_thread = threading.Thread(
        target=run_optimization, args=(study, optimization_runner, n_trials, callback)
    )
    opt_thread.start()
    while opt_thread.is_alive():
        plt.pause(0.1)
    opt_thread.join()

    if study.trials:
        logger.info(f"Optimization Completed. Best Trial: {study.best_trial.number}")
        logger.info(f"Best Score: {study.best_trial.value:.2f}")
        logger.info(f"Best Parameters: {study.best_trial.params}")
        best_config = config_manager.config
        best_config["chord_detection"]["sensitivity"] = round(
            study.best_trial.params.get("sensitivity", 0.96), 2
        )
        best_config["chord_detection"]["onset_delta"] = round(
            study.best_trial.params.get("onset_delta", 0.11), 2
        )
        best_config["chord_detection"]["hop_length"] = study.best_trial.params.get(
            "hop_length", 448
        )
        config_manager.save_config(best_config)
    else:
        logger.info("No trials were completed.")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
