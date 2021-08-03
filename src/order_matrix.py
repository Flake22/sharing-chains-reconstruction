import numpy as np
import random
import csv

def order_matrix(cm, classes):
    
    """
    This function takes in input a confusion matrix and the classes
    and returns them ordered alphabetically by reconstruction order
    """

    if len(classes)==39:
        new_order =  ['FB', 'FB-FB', 'FB-FB-FB', 'FL-FB-FB', 'TW-FB-FB', 'FL-FB', 'FB-FL-FB', 'FL-FL-FB', 'TW-FL-FB', 'TW-FB', 'FB-TW-FB', 'FL-TW-FB', 'TW-TW-FB', 'FL', 'FB-FL', 'FB-FB-FL', 'FL-FB-FL', 'TW-FB-FL', 'FL-FL', 'FB-FL-FL', 'FL-FL-FL', 'TW-FL-FL', 'TW-FL', 'FB-TW-FL', 'FL-TW-FL', 'TW-TW-FL', 'TW', 'FB-TW', 'FB-FB-TW', 'FL-FB-TW', 'TW-FB-TW', 'FL-TW', 'FB-FL-TW', 'FL-FL-TW', 'TW-FL-TW', 'TW-TW', 'FB-TW-TW', 'FL-TW-TW', 'TW-TW-TW']
    elif len(classes)==12:
        new_order = ['FB', 'FB-FB', 'FL-FB', 'TW-FB', 'FL', 'FB-FL', 'FL-FL', 'TW-FL', 'TW', 'FB-TW', 'FL-TW', 'TW-TW']
    elif len(classes)==21:
        new_order = ['FB', 'FB-FB', 'FB-FB-FB', 'FL-FB-FB', 'TW-FB-FB', 'FL-FB', 'FB-FL-FB', 'FL-FL-FB', 'TW-FL-FB', 'TW-FB', 'FL', 'FB-FL', 'FB-FB-FL', 'FL-FB-FL', 'TW-FB-FL', 'FL-FL', 'FB-FL-FL', 'FL-FL-FL', 'TW-FL-FL', 'TW-FL', 'TW']
    elif len(classes)==9:
        new_order = ['FB', 'FB-FB', 'FL-FB', 'TW-FB', 'FL', 'FB-FL', 'FL-FL', 'TW-FL', 'TW']
    else:
        return cm, classes

    mapping = {}

    for i, cl1 in enumerate(classes):
        for j, cl2 in enumerate(classes):
            mapping[cl1+'x'+cl2] = cm[i][j]

    N = len(new_order)
    new_cm = np.zeros((N,N))

    for i, cl1 in enumerate(new_order):
        for j, cl2 in enumerate(new_order):
            new_cm[i][j]=mapping[cl1+'x'+cl2]

    return new_cm, new_order
