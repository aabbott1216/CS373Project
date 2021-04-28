import matplotlib.pyplot as plt
import numpy as np


def run(sensitivity_svm, specificity_svm, sensitivity_tree, specificity_tree, title):
    sens_spec_dict_svm = dict()
    sensitivity_svm = list(sensitivity_svm)
    specificity_svm = list(specificity_svm)
    for i, key in enumerate(specificity_svm):
        sens_spec_dict_svm[key] = sensitivity_svm[i]
    sens_sorted_svm = []
    spec_sorted_svm = []
    for key in sorted(sens_spec_dict_svm):
        spec_sorted_svm.append(key)
        sens_sorted_svm.append(sens_spec_dict_svm[key])

    sens_spec_dict_tree = dict()
    sensitivity_tree = list(sensitivity_tree)
    specificity_tree = list(specificity_tree)
    for i, key in enumerate(specificity_tree):
        sens_spec_dict_tree[key] = sensitivity_tree[i]
    sens_sorted_tree = []
    spec_sorted_tree = []
    for key in sorted(sens_spec_dict_tree):
        spec_sorted_tree.append(key)
        sens_sorted_tree.append(sens_spec_dict_tree[key])

    plt.scatter(1-np.array(spec_sorted_svm), sens_sorted_svm)
    plt.scatter(1-np.array(spec_sorted_tree), sens_sorted_tree)
    plt.plot([0, 1])
    plt.legend(["Boundary", "SVM ROC", "Decision Tree ROC"])

    if (title == "kfold"):
        plot_title = "ROC Curve (K-fold)"
    elif (title == "boot"):
        plot_title = "ROC Curve (Bootstrapping)"

    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(plot_title)
    plt.show()
