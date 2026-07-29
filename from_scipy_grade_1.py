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
    metrics_model1 = [0.5313, 0.6668, 0.6230, 1-0.4998, 0.5053, 0.6065, 0.5921, 0.5909]  # SIM
    #metrics_model2 = [0.5556, 0.5515, 0.6169, 0.5603, 0.6183, 0.6698, 0.6139, 0.5886]  # SIM + ALL FX
    # metrics_model1 = [0.7831, 0.7645, 0.7421, 0.7587, 0.7603, 0.735, 0.7583, 0.7528]  # SIM + FX1 + FX2
    # metrics_model2 = [0.7448, 0.7275, 0.7442, 0.7427, 0.7392, 0.7612, 0.7415, 0.7033]  # SIM + FX1 + FX2 + FX3
    metrics_model2 = [0.4616, 0.4272, 0.5018, 0.4680, 0.4951, 0.4919, 0.5459, 0.4524]  # SIM + FX4 + FX5
    #metrics_model2 = [0.7036, 0.7146, 0.6295, 0.6777, 0.7012, 0.6146, 0.6163, 0.6856]  # ALL FX
    # metrics_model2 = [0.7351, 0.7246, 0.7046, 0.7610, 0.7424, 0.7246, 0.7504, 0.8046]  # FX1+FX2
    # metrics_model2 = [0.7819, 0.8042, 0.7669, 0.7951, 0.7886, 0.8053, 0.7757, 0.8271]  # FX1+FX2+FX3
    # metrics_model2 = [0.4451, 0.4568, 0.5530, 0.5134, 0.4874, 0.4478, 0.5166, 0.5027]  # FX4+FX5
    

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