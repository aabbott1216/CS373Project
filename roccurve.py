import matplotlib.pyplot as plt


def run(sensitivity_svm, specificity_svm, sensitivity_tree, specificity_tree):
    plt.plot(specificity_svm, sensitivity_svm)
    plt.plot(specificity_tree, sensitivity_tree)
    plt.show()
