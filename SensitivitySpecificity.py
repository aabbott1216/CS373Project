def run(actual, predicted):
    TP = 0
    FN = 0
    TN = 0
    FP = 0

    for i, val in enumerate(actual):
        if val == 1:
            if val == predicted[i]:
                TP += 1
            elif val != predicted[i]:
                FN += 1
        elif val == 0:
            if val == predicted[i]:
                TN += 1
            elif val != predicted[i]:
                FP += 1

    sensitivity = float(TP) / (TP + FN)
    specificity = float(TN) / (TN + FP)
    return (sensitivity, specificity)
