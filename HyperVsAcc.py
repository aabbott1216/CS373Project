import matplotlib.pyplot as plt

def run(hyperList, accList):
    plt.plot(hyperList, accList)
    plt.xlabel("Hyperparameter Value")
    plt.ylabel("Accuracy")
    plt.show()
