import numpy as np

def get_confusion_matrix(predictions, labels, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for i in range(len(predictions)):
        cm[labels[i]][predictions[i]] += 1

    return cm

def get_precision_recall_f1(cm):
    row_sum = cm.sum(axis = 1)
    col_sum = cm.sum(axis = 0)

    precision = np.array([cm[i][i] / x for i, x in enumerate(col_sum)])
    recall = np.array([cm[i][i] / x for i, x in enumerate(row_sum)])
    f1_score = ((2 * precision * recall) / (precision + recall))
    

    return precision.mean(), recall.mean(), f1_score.mean()
    