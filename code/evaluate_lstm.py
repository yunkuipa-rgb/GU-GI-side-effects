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
    metrics_model1 = [0.6007, 0.5081, 0.5257, 0.5771, 0.5622, 0.5589, 0.5656, 0.6156]  # SIM
    # metrics_model1 = [0.7457, 0.7206, 0.7863, 0.701, 0.7704, 0.7319, 0.7677, 0.696]  # SIM + Fx1 + Fx2
    # metrics_model1 = [0.6128, 0.6191, 0.6056, 0.5466, 0.6612, 0.6636, 0.58, 0.6772]  # SIM + All Fx
    # metrics_model3 = [0.67, 0.7275, 0.4930, 0.505, 0.6368, 0.6624, 0.7019, 0.6583]  # Model 2 metrics
    #metrics_model3 = [0.6551, 0.6881, 0.6651, 0.6625, 0.6574, 0.7212, 0.6624, 0.6039]  # SIM + Fx1 + Fx2 + Fx3
    metrics_model3 = [0.4630, 0.4653, 0.5239, 0.4866, 0.567, 0.4748, 0.54, 0.4945]  # SIM+Fx4+Fx5

    # Fx Only
    # metrics_model3 = [0.6237, 0.6289, 0.5721, 0.6492, 0.4865, 0.5651, 0.5834, 0.4943] # fx1-5
    # metrics_model3 = [0.5181, 0.5415, 0.5112, 0.5484, 0.5624, 0.4715, 0.4822, 0.4368] # fx1
    # metrics_model3 = [0.7866, 0.8289, 0.7454, 0.7672, 0.7506, 0.6966, 0.7039, 0.8248] # fx1+fx2
    # metrics_model3 = [0.6033, 0.6160, 0.6242, 0.5996, 0.6381, 0.6446, 0.6045, 0.6262] # fx1+fx2+fx3
    # metrics_model3 = [0.6251, 0.5530, 0.5513, 0.5409, 0.5230, 0.5266, 0.5901, 0.5618] # fx4+fx5

    # Compute p-value
    p_value = compute_p_values(metrics_model1, metrics_model3)

    print(f"P-value: {p_value}", np.asarray(metrics_model1).mean(), np.asarray(metrics_model1).std())
    print()

    # Interpret the result
    alpha = 0.05
    if p_value < alpha:
        print("The difference between the models is statistically significant.")
    else:
        print("The difference between the models is not statistically significant.")