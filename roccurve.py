import matplotlib.pyplot as plt


def run(sensitivity_svm, specificity_svm):
    plt.plot(1-specificity_svm, sensitivity_svm)
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    # plt.plot(specificity_tree, sensitivity_tree)
    plt.show()
