import matplotlib.pyplot as plt

def run(hyperSvm, accSvm, hyperTree, accTree):
    hyperSvmUni = []
    accSvmAvg = []
    hyperTreeUni = []
    accTreeAvg = []
    sParams = int(len(hyperSvm)/15)
    tParams = int(len(hyperTree)/15)
    
    for i in range(sParams):
        sSum = 0
        hyperSvmUni.append(hyperSvm[i*15])
        for j in range(15):
            sSum += accSvm[(i*15)+j]
        accSvmAvg.append(sSum)
        
    for i in range(tParams):
        tSum = 0
        hyperTreeUni.append(hyperTree[i*15])
        for j in range(15):
            tSum += accTree[(i*15)+j]
        accTreeAvg.append(tSum)
        
    fig1, (axis1, axis2) = plt.subplots(1, 2)
    fig1.suptitle('Hyperparameter vs. Accuracy')
    axis1.plot(hyperSvm, accSvm, color="blue", linestyle="-.")
    axis1.set_title("SVM plot")
    axis2.plot(hyperTree, accTree, color="red", linestyle="-.")
    axis2.set_title("Tree plot")
    plt.show()
