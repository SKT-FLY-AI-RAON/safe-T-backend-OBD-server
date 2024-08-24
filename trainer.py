from sklearn.metrics import confusion_matrix

def find_TPR_threshold(y, scores, desired_TPR):
    threshold = 1.0
    TPR = 0.0
    FPR = 0.0

    while TPR < desired_TPR:
        temp = scores.copy()
        temp[temp < threshold] = 0
        temp[temp >= threshold] = 1

        confusion = confusion_matrix(y, temp)
        if confusion.size != 4:
            break
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        threshold -= 0.01

    return threshold + 0.01, FPR
