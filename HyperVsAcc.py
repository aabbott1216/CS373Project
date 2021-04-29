import matplotlib.pyplot as plt


def run(hyperSvm, accSvm, hyperTree, accTree, title):

    if (title == "kfold"):
        plot_title = "Hyperparameter vs. Accuracy (K-fold)"
    elif (title == "boot"):
        plot_title = "Hyperparameter vs. Accuracy (Bootstrapping)"

    fig1, (axis1, axis2) = plt.subplots(1, 2)
    fig1.suptitle('Hyperparameter vs. Accuracy')
    plt.xscale('log')
    axis1.plot(hyperSvm, accSvm, color="blue", linestyle="-.")
    axis1.set_title("SVM plot")

    for x, y in zip(hyperSvm, accSvm):
        label = "{:.2f}".format(x)
        axis1.annotate(label,  # this is the text
                       (x, y),  # this is the point to label
                       textcoords="offset points",  # how to position the text
                       xytext=(0, -10),  # distance from text to points (x,y)
                       ha='center')  # horizontal alignment can be left, right or center

    axis2.plot(hyperTree, accTree, color="red", linestyle="-.")
    axis2.set_title("Tree plot")

    for x, y in zip(hyperTree, accTree):
        label = "{:.2f}".format(x)
        axis2.annotate(label,  # this is the text
                       (x, y),  # this is the point to label
                       textcoords="offset points",  # how to position the text
                       xytext=(0, 10),  # distance from text to points (x,y)
                       ha='center')  # horizontal alignment can be left, right or center
    plt.show()
