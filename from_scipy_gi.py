from scipy.stats import ttest_rel
import numpy as np

def compute_p_values(metrics_model1, metrics_model2):
    """
    Computes the p-value for the difference in metrics between two models using a paired t-test.

    Args:
        metrics_model1 (list): Evaluation metrics for model 1 (e.g., AUC, ACC).
        metrics_model2 (list): Evaluation metrics for model 2 (e.g., AUC, ACC).

    Returns:
        float: p-value indicating whether the differences between models are statistically significant.
    """
    if len(metrics_model1) != len(metrics_model2):
        raise ValueError("Both lists must have the same length.")

    # Perform a paired t-test
    t_stat, p_value = ttest_rel(metrics_model1, metrics_model2)

    return p_value

# Example usage
if __name__ == "__main__":
    # Metrics from two models (e.g., AUC or Accuracy scores for different datasets or folds)
    metrics_model1 = [0.5861, 0.6672, 1 - 0.4755, 0.5405, 0.5388, 0.5311, 0.5688, 0.5594]  # SIM
    # metrics_model2 = [0.6305, 0.6722, 0.6477, 0.6533, 0.7405, 0.7144, 0.7494, 0.6605]  # SIM + ALL FX
    # metrics_model2 = [0.7338, 0.7538, 0.7422, 0.7083, 0.7844, 0.7022, 0.7238, 0.6922]  # SIM + FX1 + FX2
    # metrics_model2 = [0.6205, 0.6508, 0.6638, 0.7366, 0.6383, 0.6527, 0.71, 0.6816]  # SIM + FX1 + FX2 + FX3
    metrics_model2 = [0.5394, 0.4005, 0.4394, 0.4133, 0.4905, 0.4566, 0.4388, 0.4672]  # SIM + FX4 + FX5
    # metrics_model2 = [0.5755, 0.745, 0.5683, 0.6438, 0.6288, 0.6222, 0.6461, 0.6705]  # ALL FX
    # metrics_model2 = [0.7333, 0.745, 0.7266, 0.74, 0.7377, 0.7394, 0.7611, 0.7488]  # FX1 + FX2
    # metrics_model2 = [0.7161, 0.6244, 0.6733, 0.6555, 0.6705, 0.6822, 0.6383, 0.6294]  # FX1 + FX2 + FX3
    # metrics_model2 = [0.4194, 0.5861, 0.5261, 0.3983, 0.5133, 0.5955, 0.6688, 0.5233]  # FX4 + FX5

    # Compute p-value
    p_value = compute_p_values(metrics_model1, metrics_model2)

    print(f"P-value: {p_value}", np.asarray(metrics_model2).mean(), np.asarray(metrics_model2).std())
    print()

    # Interpret the result
    alpha = 0.05
    if p_value < alpha:
        print("The difference between the models is statistically significant.")
    else:
        print("The difference between the models is not statistically significant.")
