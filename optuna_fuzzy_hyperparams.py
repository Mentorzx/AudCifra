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
    Create and return a logger with a standardized format.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_logger(__name__)


class ConfigManager:
    """
    Manages loading and saving of a YAML configuration file.
    """

    def __init__(self, config_path: str):
        """
        Initialize the ConfigManager with the given configuration file path.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> dict:
        """
        Load the YAML configuration file.

        Returns:
            dict: The configuration dictionary.

        Raises:
            SystemExit: If the configuration file does not exist.
        """
        if not os.path.exists(self.config_path):
            logger.error(f"Configuration file not found: {self.config_path}")
            sys.exit(1)
        with open(self.config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def save_config(self, config=None) -> None:
        """
        Save the configuration to the YAML configuration file.

        Args:
            config (dict, optional): Configuration dictionary to save.
                If None, the current configuration is saved.
        """
        if config is None:
            config = self.config
        with open(self.config_path, "w", encoding="utf-8") as file:
            yaml.dump(config, file, default_flow_style=False, allow_unicode=True)


class ChordExtractor:
    """
    Extracts chords from a DOCX file using a regular expression.
    """

    CHORD_PATTERN = re.compile(
        r"\b[A-Ga-g][#b]?(?:m|maj|sus|dim|aug|add)?\d*(?:/[A-Ga-g][#b]?)?\b"
    )

    @classmethod
    def extract_chords_from_docx(cls, file_path: str) -> list:
        """
        Extract chords from the specified DOCX file.

        Args:
            file_path (str): Path to the DOCX file.

        Returns:
            list: A list of extracted chords. Returns an empty list if the file does not exist.
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
    Evaluates the score by comparing generated chords with a reference.
    """

    @staticmethod
    def evaluate(reference: list, generated: list) -> float:
        """
        Evaluates the score based on the similarity between reference and generated chords.
        
        The score is computed on a scale from 0 to 100. A perfect match—where the generated chord sequence
        has the same number of chords as the reference and each chord is identical and in the same position—
        yields a score of 100. Matching chords in the same position are weighted more heavily than matching
        only the count. The final score is the product of the average fuzzy similarity (using fuzz.ratio)
        and a length factor (the ratio of the smaller to the larger list length). If either list is empty,
        the function returns -100.
        
        Parameters:
            reference (list): List of reference chords.
            generated (list): List of generated chords.
        
        Returns:
            float: The computed score (minimum 0, maximum 100), or -100 if either list is empty.
        """
        if not reference or not generated:
            logger.warning("Empty chord list detected. Returning score -100.")
            return -100.0

        n_ref = len(reference)
        n_gen = len(generated)
        L = min(n_ref, n_gen)
        total_similarity = 0.0
        for i in range(L):
            total_similarity += fuzz.ratio(reference[i], generated[i])
        avg_similarity = total_similarity / L  # Range: 0 to 100
        length_factor = float(L) / float(max(n_ref, n_gen))
        score = avg_similarity * length_factor
        logger.info(
            f"Evaluation: {n_ref} reference chords, {n_gen} generated chords, "
            f"Average Similarity: {avg_similarity:.2f}, Length Factor: {length_factor:.2f}, "
            f"Computed Score: {score:.2f}"
        )

        return max(score, 0.0)


class OptimizationRunner:
    """
    Runs the optimization process using Optuna.
    """

    def __init__(
        self, config_manager: ConfigManager, results_folder: str, reference_file: str
    ):
        """
        Initialize the OptimizationRunner.

        Args:
            config_manager (ConfigManager): Instance of ConfigManager.
            results_folder (str): Folder path where output files are stored.
            reference_file (str): Path to the reference DOCX file.
        """
        self.config_manager = config_manager
        self.results_folder = results_folder
        self.reference_file = reference_file

    def run_main_script(self) -> bool:
        """
        Execute the main.py script as a subprocess.

        Returns:
            bool: True if the script executed successfully, False otherwise.
        """
        process = subprocess.run(
            ["python", "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if process.returncode != 0:
            logger.error("Execution of main.py failed.")
            logger.error(f"Error: {process.stderr.strip()}")
            return False
        return True

    def get_latest_output_file(self) -> str:
        """
        Retrieve the latest output file from the results folder.

        Returns:
            str: Full path of the latest output file, or an empty string if none found.
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
            logger.error("No output file generated.")
            return ""
        return os.path.join(self.results_folder, files[-1])

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for the Optuna study.

        Adjusts configuration parameters, runs the main script, extracts chords from
        the output and reference files, and evaluates the resulting score.

        Args:
            trial (optuna.Trial): The current Optuna trial.

        Returns:
            float: The evaluation score for the trial.
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
        logger.info(f"Trial {trial.number}: Parameters: {config['chord_detection']}")
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
    Create a matplotlib figure with a predefined grid layout for various plots.

    Returns:
        tuple: A tuple containing the figure and a dictionary of axes.
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
    Update the score evolution plot with trial numbers and corresponding scores.

    Args:
        ax: Matplotlib Axes object.
        trial_numbers (list): List of trial numbers.
        scores (list): List of scores.
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
    """
    Sort and interpolate trial data to create a smooth curve on the plot.

    Args:
        trial_numbers (ndarray): Array of trial numbers.
        scores (ndarray): Array of scores corresponding to trial numbers.
        ax: Matplotlib Axes object.
    """
    sort_idx = np.argsort(trial_numbers)
    trial_numbers_sorted = trial_numbers[sort_idx]
    scores_sorted = scores[sort_idx]
    k = 3 if len(trial_numbers_sorted) >= 4 else 1
    try:
        xnew = np.linspace(trial_numbers_sorted.min(), trial_numbers_sorted.max(), 300)
        spl = make_interp_spline(trial_numbers_sorted, scores_sorted, k=k)
        scores_smooth = spl(xnew)
        ax.plot(xnew, scores_smooth, color="blue", label="Smoothed Curve")
    except Exception as e:
        ax.plot(trial_numbers_sorted, scores_sorted, color="blue", label="Line")
    ax.scatter(trial_numbers_sorted, scores_sorted, color="red")


def update_param_importance(ax, study):
    """
    Update the parameter importance plot based on the Optuna study.

    Args:
        ax: Matplotlib Axes object.
        study (optuna.study.Study): The Optuna study.
    """
    ax.clear()
    if len(study.trials) <= 1:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
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
    Update the parallel coordinates plot for hyperparameters.

    Args:
        ax: Matplotlib Axes object.
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
    Update slice plots that show hyperparameters versus scores.

    Args:
        axes (dict): Dictionary of matplotlib Axes.
        sv (list): List of sensitivity values.
        od (list): List of onset delta values.
        hl (list): List of hop length values.
        scores (list): List of scores.
        score_errors (list, optional): List of score error values. Defaults to 1.0 for each score.
    """
    if score_errors is None:
        score_errors = [1.0] * len(scores)
    deg = 3
    axes["slice1"].clear()
    if len(sv) > 1:
        _plot_weighted_regression(
            axes["slice1"], sv, scores, score_errors, deg=deg, color="red"
        )
    _axes_hyperparams(axes, "slice1", "Sensitivity vs. Score", "Sensitivity")
    axes["slice2"].clear()
    if len(od) > 1:
        _plot_weighted_regression(
            axes["slice2"], od, scores, score_errors, deg=deg, color="blue"
        )
    _axes_hyperparams(axes, "slice2", "Onset Delta vs. Score", "Onset Delta")
    axes["slice3"].clear()
    if len(hl) > 1:
        _plot_weighted_regression(
            axes["slice3"], hl, scores, score_errors, deg=deg, color="green"
        )
    _axes_hyperparams(axes, "slice3", "Hop Length vs. Score", "Hop Length")


def _plot_weighted_regression(ax, x_data, scores, errors, deg=3, color="red"):
    """
    Plot a weighted polynomial regression with confidence bands.

    Args:
        ax: Matplotlib Axes object.
        x_data (list): X-axis data values.
        scores (list): Y-axis data (scores).
        errors (list): Weights/errors for the data points.
        deg (int, optional): Degree of the polynomial. Defaults to 3.
        color (str, optional): Color for the plot. Defaults to "red".
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
    Configure the axes for hyperparameter plots.

    Args:
        axes (dict): Dictionary of matplotlib Axes.
        key (str): Key of the axis to configure.
        title (str): Title for the axis.
        xlabel (str): Label for the x-axis.
    """
    axes[key].set_title(title)
    axes[key].set_xlabel(xlabel)
    axes[key].set_ylabel("Score")
    axes[key].grid(True)
    axes[key].legend()


def compute_poly_error(x, cov, deg):
    """
    Compute the polynomial error for a given x value using the covariance matrix.

    Args:
        x (float): The x value.
        cov (ndarray): Covariance matrix of the polynomial fit.
        deg (int): Degree of the polynomial.

    Returns:
        float: The computed error.
    """
    basis = np.array([x**p for p in range(deg, -1, -1)])
    return np.sqrt(np.dot(basis, np.dot(cov, basis)))


def compute_confidence_band(x_values, cov, deg):
    """
    Compute the confidence band for a polynomial fit.

    Args:
        x_values (array-like): Array of x values.
        cov (ndarray): Covariance matrix from the polynomial fit.
        deg (int): Degree of the polynomial.

    Returns:
        ndarray: Array of computed errors for the confidence band.
    """
    return np.array([compute_poly_error(x, cov, deg) for x in x_values])


def update_contour_plot(ax, sv, od, scores, contour_cb):
    """
    Update the contour plot based on sensitivity, onset delta, and scores.

    Args:
        ax: Matplotlib Axes object.
        sv (list): List of sensitivity values.
        od (list): List of onset delta values.
        scores (list): List of scores.
        contour_cb: Existing colorbar for the contour plot.

    Returns:
        Updated contour colorbar.
    """
    ax.clear()
    if len(sv) <= 5:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
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
    ax.set_title("Contour (Sensitivity vs. Onset Delta)")
    ax.set_xlabel("Sensitivity")
    ax.set_ylabel("Onset Delta")
    return contour_cb


def update_heatmap(ax, sv, od, scores, heatmap_cb):
    """
    Update the heatmap plot for sensitivity, onset delta, and scores.

    Args:
        ax: Matplotlib Axes object.
        sv (list): List of sensitivity values.
        od (list): List of onset delta values.
        scores (list): List of scores.
        heatmap_cb: Existing colorbar for the heatmap plot.

    Returns:
        Updated heatmap colorbar.
    """
    ax.clear()
    sc_plot = ax.scatter(sv, od, c=scores, cmap="coolwarm", s=100)
    ax.set_title("Parameter Heatmap")
    ax.set_xlabel("Sensitivity")
    ax.set_ylabel("Onset Delta")
    if not heatmap_cb:
        heatmap_cb = plt.gcf().colorbar(sc_plot, ax=ax, label="Score (high=red)")
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
    Update all plots with the latest optimization data.

    Args:
        axes (dict): Dictionary of matplotlib Axes.
        study (optuna.study.Study): The Optuna study.
        trial (optuna.trial.FrozenTrial): The latest trial.
        contour_cb: Existing contour colorbar.
        hyper_data (dict): Dictionary containing optimization data (trial numbers, scores, hyperparameter values).

    Returns:
        Updated contour colorbar.
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
    Run the Optuna optimization process.

    Args:
        study (optuna.study.Study): The Optuna study object.
        optimization_runner (OptimizationRunner): Instance of OptimizationRunner.
        n_trials (int): Number of trials to run.
        callback_fn (function): Callback function to be executed after each trial.
    """
    study.optimize(
        optimization_runner.objective, n_trials=n_trials, callbacks=[callback_fn]
    )


def main():
    """
    Main function to execute the optimization process and update plots in real-time.

    This function sets up the configuration, optimization runner, Optuna study,
    and matplotlib plots. It runs the optimization in a separate thread, updates
    plots with each trial, and saves the best configuration at the end.
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
        fig.canvas.manager.window.wm_attributes("-topmost", 0) # type: ignore
    except Exception as e:
        logger.warning(f"Could not set window attributes: {e}")

    def on_close(event):
        global is_closing
        is_closing = True

    fig.canvas.mpl_connect("close_event", on_close)
    plt.tight_layout()

    n_trials = 150
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
        logger.info(
            f"Trial {trial.number} - Score: {trial.value:.2f} | Parameters: {trial.params}"
        )
        best = study.best_trial
        logger.info(
            f"Best Trial so far: {best.number} - Score: {best.value:.2f} | Parameters: {best.params}"
        )

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
        logger.info("No trial was completed.")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
