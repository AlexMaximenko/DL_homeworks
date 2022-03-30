import numpy as np

# Most part of this code was written as a homework on mipt-ml-base course: https://github.com/girafe-ai/ml-mipt (Neychev et al.)


# Criterion base class
class Criterion(object):
    def __init__ (self):
        self.output = None
        self.gradInput = None
        
    def forward(self, input, target):
        """
            Given an input and a target, compute the loss function 
            associated to the criterion and return the result.
            
            For consistency this function should not be overrided,
            all the code goes in `updateOutput`.
        """
        return self.updateOutput(input, target)

    def backward(self, input, target):
        """
            Given an input and a target, compute the gradients of the loss function
            associated to the criterion and return the result. 

            For consistency this function should not be overrided,
            all the code goes in `updateGradInput`.
        """
        return self.updateGradInput(input, target)
    
    def updateOutput(self, input, target):
        """
        Function to override.
        """
        return self.output

    def updateGradInput(self, input, target):
        """
        Function to override.
        """
        return self.gradInput   

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want 
        to have readable description. 
        """
        return "Criterion"

# MSECriterion
class MSECriterion(Criterion):
    def __init__(self):
        super().__init__()
        
    def updateOutput(self, input, target):   
        self.output = np.sum(np.power(input - target,2)) / input.shape[0]
        return self.output 
 
    def updateGradInput(self, input, target):
        self.gradInput  = (input - target) * 2 / input.shape[0]
        return self.gradInput

    def __repr__(self):
        return "MSECriterion"

# Negative LogLokelihood criterion
class ClassNLLCriterion(Criterion):
    def __init__(self):
        super().__init__()
        
    def updateOutput(self, input, target): 
        self.output = -np.sum(np.multiply(target, input)) / input.shape[0]
        return self.output

    def updateGradInput(self, input, target):
        self.gradInput = -target / input.shape[0]
        return self.gradInput
    
    def __repr__(self):
        return "ClassNLLCriterion"


# Hinge Loss
class HingeLoss(Criterion):
    """
    Crammer and Singer Hinge loss definition for multiclass task
    """
    def __init__(self):
        super().__init__()
        self.not_aggregated_output = None

    def updateOutput(self, input, target):
        false_margins = np.max(np.multiply(input, (1-target)), axis=1)
        true_margins = np.max(np.multiply(input, target), axis=1)
        res = np.maximum(np.ones_like(false_margins) + false_margins - true_margins, np.zeros_like(false_margins))
        self.not_aggregated_output = (res > 0).astype(np.float64)
        self.output = np.sum(res) / input.shape[0]
        return self.output

    def updateGradInput(self, input, target):
        drop_if_not_max = lambda x: (x >= np.max(x)).astype(np.float64)
        res = - target.astype(np.float64)
        res += np.apply_along_axis(drop_if_not_max, 1, np.multiply(input, (1-target)))
        res = np.diag(self.not_aggregated_output) @ res
        self.gradInput = res / input.shape[0]
        return self.gradInput

    def __repr__(self):
        return 'HingeLoss'