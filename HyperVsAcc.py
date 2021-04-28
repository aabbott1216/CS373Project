import matplotlib.pyplot as plt


def run(hyperSvm, accSvm, hyperTree, accTree, title):

    if (title == "kfold"):
        plot_title = "Hyperparameter vs. Accuracy (K-fold)"
    elif (title == "boot"):
        plot_title = "Hyperparameter vs. Accuracy (Bootstrapping)"

    fig1, (axis1, axis2) = plt.subplots(1, 2)
    fig1.suptitle(plot_title)
    axis1.plot(hyperSvm, accSvm, color="blue", linestyle="-.")
    axis1.set_title("SVM plot")
    axis2.plot(hyperTree, accTree, color="red", linestyle="-.")
    axis2.set_title("Tree plot")
    plt.show()
