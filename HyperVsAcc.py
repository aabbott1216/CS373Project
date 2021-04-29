import matplotlib.pyplot as plt


def run(hyperSvm, accSvm, hyperTree, accTree, title):

    if (title == "kfold"):
        plot_title = "Hyperparameter vs. Accuracy (K-fold)"
    elif (title == "boot"):
        plot_title = "Hyperparameter vs. Accuracy (Bootstrapping)"

    fig1, (axis1, axis2) = plt.subplots(1, 2)
    fig1.suptitle('Hyperparameter vs. Accuracy')
    axis1.plot(hyperSvm, accSvm, color="blue", linestyle="-.")
    axis1.set_title("SVM plot")

    for x, y in zip(hyperSvm, accSvm):
        label = "{:.1f}".format(x)
        axis1.annotate(label,
                       (x, y),
                       textcoords="offset points",
                       xytext=(15, -20),
                       ha='center',
                       arrowprops=dict(arrowstyle="->"))

    axis2.plot(hyperTree, accTree, color="red", linestyle="-.")
    axis2.set_title("Tree plot")

    for x, y in zip(hyperTree, accTree):
        label = "{:.2f}".format(x)
        axis2.annotate(label,
                       (x, y),
                       textcoords="offset points",
                       xytext=(15, 20),
                       ha='center',
                       arrowprops=dict(arrowstyle="->"))
    plt.show()
