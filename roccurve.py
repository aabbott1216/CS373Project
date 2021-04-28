import matplotlib.pyplot as plt
import numpy as np


def run(sensitivity_svm, specificity_svm):
    sens_spec_dict = dict()
    sensitivity_svm = list(sensitivity_svm)
    specificity_svm = list(specificity_svm)
    for i, key in enumerate(specificity_svm):
        sens_spec_dict[key] = sensitivity_svm[i]
    sens_sorted = []
    spec_sorted = []
    for key in sorted(sens_spec_dict):
        spec_sorted.append(key)
        sens_sorted.append(sens_spec_dict[key])
    plt.Line2D(1-np.array(spec_sorted), sens_sorted)
    plt.plot([0, 1])

    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title("ROC curve")
    # plt.plot(specificity_tree, sensitivity_tree)
    plt.show()
