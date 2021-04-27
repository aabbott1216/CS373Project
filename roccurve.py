import matplotlib.pyplot as plt

def run(sensitivity_svm, specificity_svm, sensitivity_tree, specificity_tree):
    for i in range(len(specificity_svm)):
        specificity_svm[i]=1-specificity_svm[i]
    for i in range(len(specificity_tree)):
        specificity_tree[i]=1-specificity_tree[i]
    plt.plot(specificity_svm, sensitivity_svm, color="red", label="SVM")
    plt.plot(specificity_tree, sensitivity_tree, color="blue", label="Decision Tree")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.legend()
    plt.show()
