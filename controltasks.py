import copy
from collections import defaultdict 
import numpy as np 
#
def control_y(data_x, data_y, flattened, dist):
    """
    Function to create control data for the POS tagging task based on a distribution
    """
    control_dict = defaultdict(lambda:None)
    
    # Retrieve 'POS tags' and distribution values
    keys = np.array(list(dist.keys()))
    values = np.array(list(dist.values()))
    
    # Normalize to get probabilities and sample keys
    probs = values / np.sum(values)
    
    # Make dict and get a full new y based on dict
    control_dict = {word.lower():np.random.choice(keys,replace=True,p=probs) for word in flattened if not word.lower() in control_dict}
    new_y = np.array([control_dict[word.lower()] for word in flattened])
    
    return new_y, control_dict

#
def dev_test_y(control_dict_train, flattened, dist):
    """
    Create test and dev control sets based on word mappings from train dict
    """
    # Retrieve 'POS tags' and distribution values
    keys = np.array(list(dist.keys()))
    values = np.array(list(dist.values()))
    
    # Normalize to get probabilities and sample keys
    probs = values / np.sum(values)
    
    # Initialize control dict and new y
    new_y = []
    control_dict = copy.deepcopy(control_dict_train)
    
    #
    for word in flattened:
        word = word.lower()
        if word not in control_dict:
            control_dict[word] = np.random.choice(keys,replace=True,p=probs)

    new_y = np.array([control_dict[word.lower()] for word in flattened])
    
    return np.array(new_y)