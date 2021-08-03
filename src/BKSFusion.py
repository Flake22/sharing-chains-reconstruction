import numpy as np
from itertools import product
import pandas as pd
from matplotlib import pyplot as plt


"""
Behavior-Knowledge-Space (BKS) Fusion of Classifiers 

@author: Sebastiano Verde, PhD

University of Trento
sebastiano.verde@unitn.it
""" 


class BKSFuser:

    """
    BKS Fuser

    This class defines a BKS object that can be used to fuse the outputs of 
    multiple classifiers. 
    The BKS first needs to be trained on a different set of samples to that 
    used to train the classifiers (e.g., if the original dataset was split 
    into train-validation-test, validation data can serve the purpose).
    """ 


    # Constants
    _REJECT_LABEL = 'REJ'#-1
    _SOFT_OUT = 'soft'
    _HARD_OUT = 'hard'


    def __init__(self, num_classifiers, num_classes, name_classes):

        self._num_classifiers = num_classifiers        # No. classifiers, K
        self._num_classes = num_classes                # No. classes, M
        self._name_classes = list(name_classes)

        # Generate the set of M^K classification tuples: (c_1, c_2, ... c_K)
        self._class_tuples = list(product(
            [i for i in self._name_classes],     # [0, 1, ..., M]
            repeat=self._num_classifiers))

        # Lookup table, M-by-M^K matrix, initialized to all zeros
        self._lookup_table = np.zeros((self._num_classes,
                                       len(self._class_tuples)))
        
        # Mark current instance as untrained
        self._is_trained = False


    """
    Train

    Receives a list of lists of classifiers outputs calculated on a given 
    dataset, and the associated list of true labels.
    Populates the BKS lookup table by counting the number of samples for which
    each tuple of classifiers outputs gets associated to each class.
    """

    def train(self, classifiers_outputs, true_labels):

        # Check input
        self._check_classifiers(classifiers_outputs)
        self._check_labels(true_labels)

        # Populate lookup table
        for i, label in enumerate(true_labels):
            class_tuple = []
            for output in classifiers_outputs:
                class_tuple.append(output[i])

            # Find column index related to current output combination
            column_idx = self._class_tuples.index(tuple(class_tuple))

            # Increment cell in lookup table
            self._lookup_table[self._name_classes.index(label)][column_idx] += 1

        # Mark as trained
        self._is_trained = True


    """
    Fuse

    Receives a list of lists of classifiers outputs. Provides the fusion of 
    such decisions according to the trained lookup table. Can return both hard
    decisions (default) or soft probability values over the output classes.
    """

    def fuse(self, classifiers_outputs, output_type='hard'):
        # Check input
        self._check_training()
        self._check_classifiers(classifiers_outputs)
        self._check_output_type(output_type)
        
        # Initialize soft output values
        predictions = np.zeros((len(classifiers_outputs),
                            self._num_classes))
        # Initialize hard output labels and probabiliteis
        decisions = []
        probs = []
        
        # Initialize rejection distribution
        self._last_rejection_distribution = np.zeros(len(self._class_tuples))

        # Loop over classifiers' outputs
        for i in range(len(classifiers_outputs)):
            
            # Reject if tuple contains rejection value
            class_tuple = tuple(classifiers_outputs)
            if any(np.array(class_tuple) == self._REJECT_LABEL):
            #if self._REJECT_LABEL in np.array(class_tuple):
                
                predictions[i,:] = float('NaN')
                decisions[i] = self._REJECT_LABEL
                
            else:        
                # Find column index related to current output combination
                column_idx = self._class_tuples.index(class_tuple)
                
                # Soft prediction (set to NaN if column is empty)
                column_sum = np.sum(self._lookup_table[:,column_idx])
                if not column_sum == 0:
                    predictions[i,:] = (self._lookup_table[:,column_idx] /
                                        column_sum)
                else:
                    predictions[i,:] = float('NaN')
                        
                # Check if prediction is ambiguous
                if self._is_ambiguous(predictions[i,:]):
    
                    # Reject sample
                    decisions.append(self._REJECT_LABEL)
                    probs.append(float('NaN'))
                    self._last_rejection_distribution[column_idx] += 1

                else:
                    # Output label and probabilites
                    decisions.append(self._name_classes[np.argmax(predictions[i,:])])
                    probs.append(max(predictions[i,:]))
                    

        if output_type == self._HARD_OUT: 

            return decisions, probs

        elif output_type == self._SOFT_OUT:

            return predictions


    """
    Get lookup table

    Returns the BKS lookup table as a 'DataFrame' object.
    """

    def get_lookup_table(self):

        return pd.DataFrame(self._lookup_table, 
                            columns=self._class_tuples)
    
    
    """
    Input distribution

    Analyze the distribution of input samples (i.e., classifiers' outputs) over
    the set of class tuples. Display a histogram of the samples distribution, 
    along with a second one from the samples in the lookup table.
    """
    
    def input_distribution(self, classifiers_outputs):
        
        self._check_classifiers(classifiers_outputs)
        
        # Initialize array of input sample distribution
        input_distrib = np.zeros(len(self._class_tuples))

        # Loop over classifiers' outputs
        for i in range(len(classifiers_outputs[0])):
            table_idx = []
            for output in classifiers_outputs:
                table_idx.append(output[i])

            # Find column index related to current output combination
            column_idx = self._class_tuples.index(tuple(table_idx))
            
            # Update sample distribution
            input_distrib[column_idx] += 1
        
        # Distribution of samples in the lookup table
        train_distrib = np.sum(self._lookup_table.T, axis=1)

        # Create DataFrame for displaying plot
        distrib_df = pd.DataFrame(np.vstack((train_distrib, input_distrib)).T, 
                    columns=['Fusion training',
                             'Input samples'], 
                    index=self._class_tuples)
        distrib_df.plot.bar(figsize=(12,4)) 
        plt.title("Distribution of samples over the set of classification tuples")
        plt.xlabel('Class tuples')
        plt.ylabel('No. samples')
        plt.tick_params(labelsize='small')
        plt.grid()
    
    
    """
    Rejections distribution

    Analyze the distribution of samples that were rejected during the last 
    performed fusion, over the set of class tuples. Display a histogram of the
    samples distribution, along with a second one from the samples in the 
    lookup table.
    """

    def rejection_distribution(self):
        
        # Create DataFrame for displaying plot
        distrib_df = pd.DataFrame(self._last_rejection_distribution.T, 
                    columns=['Rejected samples'], 
                    index=self._class_tuples)
        distrib_df.plot.bar(figsize=(12,4)) 
        plt.title("Distribution of rejected samples over the set of classification tuples")
        plt.xlabel('Class tuples')
        plt.ylabel('No. samples')
        plt.tick_params(labelsize='small')
        plt.grid()


    """
    Remove rejections
    
    Discard rejected samples from predicted and true labels.
    """

    def remove_rejections(self, predictions, true_labels):
        
        new_predictions = []
        new_true_labels = []
        
        for i, pred in enumerate(predictions):
            if not pred == self._REJECT_LABEL:
                new_predictions.append(pred)
                new_true_labels.append(true_labels[i])
                
        return new_predictions, new_true_labels


    """
    Private methods
    """
    
    def _check_classifiers(self, classifiers_outputs):
        if len(classifiers_outputs) != self._num_classifiers:
            raise ValueError('Found unexpected number of classifiers: %d' %
                             (len(classifiers_outputs)))


    def _check_labels(self, true_labels):

        if len(set(true_labels)) != self._num_classes:
            raise ValueError('Found unexpected number of label values: %d' %
                             (len(set(true_labels))))


    def _check_training(self):

        if not self._is_trained:
            raise AttributeError("'BKSFusion' object must be trained before calling 'fuse'")


    def _check_output_type(self, output_type):

        if (not output_type == self._SOFT_OUT and
            not output_type == self._HARD_OUT):
            raise ValueError('Found invalid output_type: ' + output_type)


    def _is_ambiguous(self, prediction):
        
        # Prediction contains one and only one maximum value
        if np.count_nonzero(prediction == np.max(prediction)) == 1:
            return False
        else:
            return True
