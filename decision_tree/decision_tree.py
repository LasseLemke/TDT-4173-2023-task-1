import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class DecisionTree:
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leafs=1):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.model = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leafs = min_samples_leafs

    def __repr__(self):
        return str(f'DecisionTree(max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, min_samples_leafs={self.min_samples_leafs})')

    def show_model(self):
        return self.model.__repr__()
    def fit(self, X, y):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        self.model = Node(None)
        self.model.fit(X, y, remaining_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leafs=self.min_samples_leafs)

    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """
        self.model.reset_n_examples()
        return X.apply(self.model.predict, axis=1)
    
    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        return self.model.get_rules()


# --- Some utility functions 
    
def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))

def oneHotEncode(df,y):
    for col in df.drop(y,axis=1).columns:
        if df[col].dtype == 'object':
            df = pd.concat([df.drop(columns=[col]), pd.get_dummies(df[col], prefix=col)], axis=1)
    return df


class Node:
    def __init__(self, parent):
        self.parent = parent
        self.children = {}
        self.split_attr = None
        self.label = None
        self.n_examples = 0

    def fit(self, X, y, remaining_depth=None, min_samples_split=2, min_samples_leafs=1):
        self.label = pd.value_counts(y).idxmax()
        if remaining_depth is not None:
            if remaining_depth == 0:
                return
            else:
                remaining_depth -= 1

        if X.shape[0] < min_samples_split:
            return
            
        if len(np.unique(y)) == 1:
            return
        
        if X.shape[1] == 0:
            return
        
        ent = entropy(pd.value_counts(y).to_numpy())
        best_attr = self._get_best_split(X, y, ent)

        if best_attr is None:
            return

        self.split_attr = best_attr

        for val in np.unique(X[best_attr]):
            y_child = y[X[best_attr] == val]
            X_child = X[X[best_attr] == val].drop(columns=[best_attr])

            if y_child.shape[0] < min_samples_leafs:
                continue

            self.children[val] = Node(self)
            self.children[val].fit(X_child,y_child, remaining_depth=remaining_depth, min_samples_split=min_samples_split, min_samples_leafs=min_samples_leafs)
        

            
    def reset_n_examples(self):
        self.n_examples = 0
        for child in self.children.values():
            child.reset_n_examples()

    def predict(self, X):
        self.n_examples += 1
        if self.split_attr is None:
            return self.label
        else: 
            if X[self.split_attr] not in self.children:
                return self.label
            else:
                return self.children[X[self.split_attr]].predict(X)
    
    def get_rules(self):
        if self.split_attr is None:
            return [([], self.label)]
        else:
            rules = []
            for val, child in self.children.items():
                rules += [([(self.split_attr, val)] + r, l) for r, l in child.get_rules()]
            return rules

    def _get_best_split(self, X, y, ent):
        best_attr = None
        highest_ig = 0

        for attr_name, attr_values in X.items():
            ents = [entropy(y[attr_values==attr_value].value_counts().to_numpy()) for attr_value in attr_values.unique()]
            probs = [y[attr_values==attr_value].shape[0] for attr_value in attr_values.unique()]
            ig = ent - np.dot(ents,probs)/np.sum(probs)
            if ig > highest_ig:
                best_attr = attr_name
                highest_ig = ig
        
        return best_attr


    def __repr__(self, level=0):
        if self.split_attr is None:
            #return str('\t'*level + '-> ' + self.label + str(self.n_examples) + '\n')
            return f'{self.split_attr}, --> {self.label} ({self.n_examples})\n'
        
        else:
            s = f'{self.split_attr}==, --> {self.label} ({self.n_examples})\n'
            for val, child in self.children.items():
                s += str('\t'*(level+1) + str(val) + ',')
                s += child.__repr__(level+1)
            return s
    
    # value, split_attr, label, n_examples