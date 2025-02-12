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
from scipy.interpolate import make_interp_spline

contour_colorbar = None
heatmap_colorbar = None


def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a logger with a standardized format.

    Parameters:
        name (str): Name for the logger.

    Returns:
        logging.Logger: A configured logger.
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

        Returns:
            dict: The configuration dictionary.
        """
        if not os.path.exists(self.config_path):
            logger.error(f"Configuration file not found: {self.config_path}")
            sys.exit(1)
        with open(self.config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def save_config(self, config) -> None:
        """
        Saves the given configuration (or the current configuration if None) to the YAML file.

        Parameters:
            config (dict, optional): The configuration to save. Defaults to None.
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

        Parameters:
            file_path (str): The path to the DOCX file.

        Returns:
            list: A list of chord strings.
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
    Computes a score by comparing generated chords with a reference list.

    The score is computed using the formula:

        score = (3 * number of correct matches) - (2 * false positives) - (2 * false negatives)

    A correct match is counted if the fuzzy similarity between a reference chord and a generated chord exceeds 75.
    This function also logs:
      - The full lists of chords compared.
      - The number of correct matches.
      - The number of false positives.
      - The number of false negatives.
      - The final computed score.
    If either list is empty, -100 is returned.
    """

    @staticmethod
    def evaluate(reference: list, generated: list) -> float:
        """
        Compares the reference and generated chord lists and returns a score.

        Parameters:
            reference (list): List of reference chords.
            generated (list): List of generated chords.

        Returns:
            float: The computed score. Returns -100 if either list is empty.
        """
        if not reference or not generated:
            return -100.0
        ratios = [fuzz.ratio(r, g) for r, g in zip(reference, generated)]
        avg_ratio = sum(ratios) / len(ratios) if ratios else 0
        false_positives = max(0, len(generated) - len(reference))
        false_negatives = max(0, len(reference) - len(generated))
        score = avg_ratio - false_positives - false_negatives
        logger.info(f"Reference chords: {reference}")
        logger.info(f"Generated chords: {generated}")
        logger.info(
            f"Score Breakdown - Correct: {avg_ratio}, False Positives: {false_positives}, False Negatives: {false_negatives}"
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

        Returns:
            bool: True if the process completes successfully, False otherwise.
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
        Retrieves the most recently modified file from the results folder.

        Returns:
            str: The path to the latest output file, or an empty string if none exist.
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

        Randomly suggests values for hyperparameters (Sensitivity: [0,1], Onset Delta: [0,1],
        Hop Length: multiples of 64 between 64 and 512), updates the configuration, runs the main process,
        and returns the computed score. If the main process fails or no output file is generated, returns -100.

        Parameters:
            trial (optuna.trial.FrozenTrial): The current trial object.

        Returns:
            float: The computed score.
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
    Creates and returns the matplotlib figure and a dictionary of axes for the various plots.

    The layout consists of:
      - Row 1: Score Evolution, Parameter Importance, Parallel Coordinates.
      - Row 2: Three slice plots (Sensitivity vs Score, Onset Delta vs Score, Hop Length vs Score),
               a Contour Plot (Sensitivity vs Onset Delta), and a Hyperparameter Heatmap.

    Returns:
        tuple: (fig, axes) where fig is the matplotlib Figure and axes is a dictionary of Axes objects.
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


def update_score_evolution(ax, trial_numbers, scores):
    """
    Updates the Score Evolution plot with the latest trial data.

    Parameters:
        ax (matplotlib.axes.Axes): The axes object for the score evolution plot.
        trial_numbers (list): List of trial numbers.
        scores (list): List of corresponding score values.
    """
    ax.clear()
    trial_numbers = np.array(trial_numbers)
    scores = np.array(scores)

    if len(trial_numbers) > 1:
        _sort_interpolation(trial_numbers, scores, ax)
    else:
        ax.plot(trial_numbers, scores, marker="o", color="blue")
    ax.set_title("Score Evolution")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Score")
    ax.grid(True)
    ax.legend()


def _sort_interpolation(trial_numbers, scores, ax):
    sort_idx = np.argsort(trial_numbers)
    trial_numbers_sorted = trial_numbers[sort_idx]
    scores_sorted = scores[sort_idx]
    k = 3 if len(trial_numbers_sorted) >= 4 else 1

    try:
        xnew = np.linspace(trial_numbers_sorted.min(), trial_numbers_sorted.max(), 300)
        spl = make_interp_spline(trial_numbers_sorted, scores_sorted, k=k)
        scores_smooth = spl(xnew)
        ax.plot(xnew, scores_smooth, color="blue", label="Curva Suave")
    except Exception as e:
        ax.plot(trial_numbers_sorted, scores_sorted, color="blue", label="Linha")
    ax.scatter(trial_numbers_sorted, scores_sorted, color="red")


def update_param_importance(ax, study):
    """
    Updates the Parameter Importance plot with the importance computed by Optuna.

    If there is insufficient trial data or no importance computed, displays a message.

    Parameters:
        ax (matplotlib.axes.Axes): The axes object for the parameter importance plot.
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
    Updates the Parallel Coordinates plot with normalized hyperparameter values.

    Sensitivity and Onset Delta are assumed to be in [0,1]. Hop Length is normalized from [64,512].

    Parameters:
        ax (matplotlib.axes.Axes): The axes for the parallel coordinates plot.
        trial_numbers (list): List of trial numbers.
        sv (list): List of sensitivity values.
        od (list): List of onset delta values.
        hl (list): List of hop length values.
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


def update_slice_plots(axes, sv, od, hl, scores, score_errors=None):
    """
    Updates the slice plots showing the relationship between each hyperparameter and the score.

    For each plot (Sensitivity vs. Score, Onset Delta vs. Score, and Hop Length vs. Score):
      - Data points are plotted with error bars.
      - A weighted polynomial regression (degree 2) is performed using the errors.
      - A regression curve with a confidence band (calculated via error propagation) is drawn.

    Parameters:
        axes (dict): Dictionary containing the axes for "slice1", "slice2", and "slice3".
        sv (list): List of sensitivity values.
        od (list): List of onset delta values.
        hl (list): List of hop length values.
        scores (list): List of score values.
        score_errors (list, optional): List of errors associated with each score.
                                       If not provided, a constant error of 1 is used for all points.
    """
    if score_errors is None:
        score_errors = [1.0] * len(scores)
    deg = 3

    # Plot 1: Sensitivity vs. Score
    axes["slice1"].clear()
    if len(sv) > 1:
        _plot_weighted_regression(
            axes["slice1"], sv, scores, score_errors, deg=deg, color="red"
        )
    _axes_hyperparams(axes, "slice1", "Sensitivity vs. Score", "Sensitivity")

    # Plot 2: Onset Delta vs. Score
    axes["slice2"].clear()
    if len(od) > 1:
        _plot_weighted_regression(
            axes["slice2"], od, scores, score_errors, deg=deg, color="blue"
        )
    _axes_hyperparams(axes, "slice2", "Onset Delta vs. Score", "Onset Delta")

    # Plot 3: Hop Length vs. Score
    axes["slice3"].clear()
    if len(hl) > 1:
        _plot_weighted_regression(
            axes["slice3"], hl, scores, score_errors, deg=deg, color="green"
        )
    _axes_hyperparams(axes, "slice3", "Hop Length vs. Score", "Hop Length")


def _plot_weighted_regression(ax, x_data, scores, errors, deg=3, color="red"):
    """
    Performs a weighted polynomial regression on the provided data and plots the data points,
    regression curve, and its confidence band on the given axis.

    The function sorts the x_data, applies a weighted polynomial regression (using np.polyfit),
    and calculates the confidence band using error propagation through the covariance matrix.

    Parameters:
        ax (matplotlib.axes.Axes): The axis on which to plot.
        x_data (list or array-like): The hyperparameter values to be used as x-coordinates.
        scores (list or array-like): The score values corresponding to each x_data value.
        errors (list or array-like): The error values associated with each score.
        deg (int, optional): The degree of the polynomial to fit. Defaults to 3.
        color (str, optional): The color used for the plot elements. Defaults to "red".

    Returns:
        None
    """
    x_arr = np.array(x_data)
    scores_arr = np.array(scores)
    errors_arr = np.array(errors)
    sort_idx = np.argsort(x_arr)
    x_sorted = x_arr[sort_idx]
    scores_sorted = scores_arr[sort_idx]
    errors_sorted = errors_arr[sort_idx]
    try:
        fit_params, cov = np.polyfit(
            x_sorted, scores_sorted, deg=deg, w=1.0 / errors_sorted, cov=True
        )
        fit_poly = np.poly1d(fit_params)
        x_fit = np.linspace(x_sorted.min(), x_sorted.max(), 300)
        y_fit = fit_poly(x_fit)
        y_fit_err = compute_confidence_band(x_fit, cov, deg)
        ax.errorbar(
            x_sorted,
            scores_sorted,
            yerr=errors_sorted,
            fmt="o",
            capsize=5,
            color=color,
            label="Data",
        )
        ax.plot(x_fit, y_fit, color=color, label=f"Weighted Regression (deg={deg})")
        ax.fill_between(
            x_fit,
            y_fit - y_fit_err,
            y_fit + y_fit_err,
            color=color,
            alpha=0.2,
            label="Confidence Band",
        )
    except Exception as e:
        ax.scatter(x_sorted, scores_sorted, color=color, label="Data")


def _axes_hyperparams(axes, key, title, xlabel):
    """
    Configures the axis attributes (title, labels, grid, and legend) for the hyperparameter plot.

    Parameters:
        axes (dict): Dictionary containing the axes.
        key (str): Key of the axis to be configured (e.g., "slice1").
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
    """
    axes[key].set_title(title)
    axes[key].set_xlabel(xlabel)
    axes[key].set_ylabel("Score")
    axes[key].grid(True)
    axes[key].legend()


def compute_poly_error(x, cov, deg):
    """
    Computes the error propagation for a polynomial of given degree at value x.

    Parameters:
        x (float): The point at which to evaluate the error.
        cov (ndarray): Covariance matrix of the polynomial coefficients.
        deg (int): Degree of the polynomial.

    Returns:
        float: The propagated error at x.
    """
    basis = np.array([x**p for p in range(deg, -1, -1)])
    return np.sqrt(np.dot(basis, np.dot(cov, basis)))


def compute_confidence_band(x_values, cov, deg):
    """
    Computes the confidence band (errors) for an array of x values.

    Parameters:
        x_values (array-like): Points at which to evaluate the error.
        cov (ndarray): Covariance matrix of the polynomial coefficients.
        deg (int): Degree of the polynomial.

    Returns:
        ndarray: Array of propagated errors corresponding to each x.
    """
    return np.array([compute_poly_error(x, cov, deg) for x in x_values])


def update_contour_plot(ax, sv, od, scores, contour_cb):
    """
    Updates the contour plot (Sensitivity vs. Onset Delta) based on the provided data.

    If fewer than 6 data points exist, displays a "Not enough data" message.
    Otherwise, draws a contour plot using tricontourf and updates the existing colorbar.

    Parameters:
        ax (matplotlib.axes.Axes): The axes for the contour plot.
        sv (list): List of sensitivity values.
        od (list): List of onset delta values.
        scores (list): List of score values.
        contour_cb: Existing contour colorbar object (or None).

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
        if not contour_cb:
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
    Updates the hyperparameter heatmap.

    Displays a scatter plot with Sensitivity on the x-axis and Onset Delta on the y-axis.
    The point color (using the "coolwarm" colormap) represents the score, where red indicates higher scores
    and blue indicates lower scores. Updates the existing colorbar if available.

    Parameters:
        ax (matplotlib.axes.Axes): The axes for the heatmap.
        sv (list): List of sensitivity values.
        od (list): List of onset delta values.
        scores (list): List of score values.
        heatmap_cb: Existing heatmap colorbar object (or None).

    Returns:
        The updated heatmap colorbar object.
    """
    ax.clear()
    sc_plot = ax.scatter(sv, od, c=scores, cmap="coolwarm", s=100)
    ax.set_title("Hyperparameter Heatmap")
    ax.set_xlabel("Sensitivity")
    ax.set_ylabel("Onset Delta")
    if not heatmap_cb:
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
    Updates all plots with the latest trial data and logs key evaluation details.

    The hyper_data dictionary must contain the following keys:
      'trial_numbers', 'scores', 'sensitivity_values', 'onset_delta_values', 'hop_length_values', 'times'.

    This function sequentially updates:
      - The Score Evolution plot.
      - The Parameter Importance plot.
      - The Parallel Coordinates plot.
      - The three slice plots (individual scatter plots for each hyperparameter vs. score).
      - The Contour Plot (Sensitivity vs. Onset Delta) and updates its colorbar.
      - The Hyperparameter Heatmap (scatter plot where the color represents the score, using the "coolwarm" colormap).

    The ScoreEvaluator.evaluate function (called elsewhere) logs the full lists compared as well as the number
    of correct matches, false positives, and false negatives.

    Parameters:
        axes (dict): Dictionary of matplotlib axes for each plot.
        study (optuna.study.Study): The Optuna study object.
        trial (optuna.trial.FrozenTrial): The most recent trial.
        contour_cb: The existing contour colorbar object (or None).
        hyper_data (dict): A dictionary containing trial data lists.

    Returns:
        The updated contour colorbar object.
    """
    global heatmap_colorbar
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
    heatmap_colorbar = update_heatmap(axes["heatmap"], sv, od, sc, heatmap_colorbar)

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
