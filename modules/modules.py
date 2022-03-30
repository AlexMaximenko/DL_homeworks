import numpy as np

# Most part of this code was written as a homework on mipt-ml-base course: https://github.com/girafe-ai/ml-mipt (Neychev et al.)


# Module is an abstract class which defines fundamental methods necessary for a training a neural network. You do not need to change anything here, just read the comments.
class Module(object):
    """
    Basically, you can think of a module as of a something (black box) 
    which can process `input_` data and produce `ouput` data.
    This is like applying a function which is called `forward`: 
        
        output = module.forward(input_)
    
    The module should be able to perform a backward pass: to differentiate the `forward` function. 
    More, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule. 
    
        gradInput = module.backward(input_, gradOutput)
    """
    def __init__ (self):
        self.output = None
        self.gradInput = None
        self.training = True
    
    def forward(self, input_):
        """
        Takes an input_ object, and computes the corresponding output of the module.
        """
        return self.updateOutput(input_)

    def backward(self,input_, gradOutput):
        """
        Performs a backpropagation step through the module, with respect to the given input_.
        
        This includes 
         - computing a gradient w.r.t. `input_` (is needed for further backprop),
         - computing a gradient w.r.t. parameters (to update parameters while optimizing).
        """
        self.updateGradInput(input_, gradOutput)
        self.accGradParameters(input_, gradOutput)
        return self.gradInput
    

    def updateOutput(self, input_):
        """
        Computes the output using the current parameter set of the class and input_.
        This function returns the result which is stored in the `output` field.
        
        Make sure to both store the data in `output` field and return it. 
        """
        pass

    def updateGradInput(self, input_, gradOutput):
        """
        Computing the gradient of the module with respect to its own input_. 
        This is returned in `gradInput`. Also, the `gradInput` state variable is updated accordingly.
        
        The shape of `gradInput` is always the same as the shape of `input_`.
        
        Make sure to both store the gradients in `gradInput` field and return it.
        """
        pass   
    
    def accGradParameters(self, input_, gradOutput):
        """
        Computing the gradient of the module with respect to its own parameters.
        No need to override if module has no parameters (e.g. ReLU).
        """
        pass
    
    def zeroGradParameters(self): 
        """
        Zeroes `gradParams` variable if the module has params.
        """
        pass
        
    def getParameters(self):
        """
        Returns a list with its parameters. 
        If the module does not have parameters return empty list. 
        """
        return []
        
    def getGradParameters(self):
        """
        Returns a list with gradients with respect to its parameters. 
        If the module does not have parameters return empty list. 
        """
        return []
    
    def train(self):
        """
        Sets training mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True
    
    def evaluate(self):
        """
        Sets evaluation mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = False
    
    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want 
        to have readable description. 
        """
        return "Module"


# Sequential container
class Sequential(Module):
    """
         This class implements a container, which processes `input_` data sequentially. 
         
         `input_` is processed by each module (layer) in self.modules consecutively.
         The resulting array is called `output`. 
    """
    
    def __init__ (self):
        super(Sequential, self).__init__()
        self.modules = []
   
    def add(self, module):
        """
        Adds a module to the container.
        """
        self.modules.append(module)

    def updateOutput(self, input_):
        """
        Basic workflow of FORWARD PASS:
        
            y_0    = module[0].forward(input_)
            y_1    = module[1].forward(y_0)
            ...
            output = module[n-1].forward(y_{n-2})   
            
            
        Just write a little loop. 
        """

        self.output = input_

        for module in self.modules:
            self.output = module.forward(self.output)

        return self.output

    def backward(self, input_, gradOutput):
        """
        Workflow of BACKWARD PASS:
            
            g_{n-1} = module[n-1].backward(y_{n-2}, gradOutput)
            g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
            ...
            g_1 = module[1].backward(y_0, g_2)   
            gradInput = module[0].backward(input_, g_1)   
             
             
        !!!
                
        To ech module you need to provide the input_, module saw while forward pass, 
        it is used while computing gradients. 
        Make sure that the input_ for `i-th` layer the output of `module[i]` (just the same input_ as in forward pass) 
        and NOT `input_` to this Sequential module. 
        
        !!!
        
        """
        for i in range(len(self.modules)-1, 0, -1):
            gradOutput = self.modules[i].backward(self.modules[i-1].output, gradOutput)

        self.gradInput = self.modules[0].backward(input_, gradOutput)

        return self.gradInput
      

    def zeroGradParameters(self): 
        for module in self.modules:
            module.zeroGradParameters()
    
    def getParameters(self):
        """
        Should gather all parameters in a list.
        """
        return [x.getParameters() for x in self.modules]
    
    def getGradParameters(self):
        """
        Should gather all gradients w.r.t parameters in a list.
        """
        return [x.getGradParameters() for x in self.modules]
    
    def __repr__(self):
        string = "".join([str(x) + '\n' for x in self.modules])
        return string
    
    def __getitem__(self,x):
        return self.modules.__getitem__(x)
    
    def train(self):
        """
        Propagates training parameter through all modules
        """
        self.training = True
        for module in self.modules:
            module.train()
    
    def evaluate(self):
        """
        Propagates training parameter through all modules
        """
        self.training = False
        for module in self.modules:
            module.evaluate()

# Linear transform layer
class Linear(Module):
    """
    A module which applies a linear transformation 
    A common name is fully-connected layer, InnerProductLayer in caffe. 
    
    The module should work with 2D input of shape (n_samples, n_feature).
    """
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()
       
        stdv = 1./np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size = (n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size = n_out)
        
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)
        
    def updateOutput(self, input):
        self.output = input @ self.W.T + self.b
        return self.output
    
    def updateGradInput(self, input, gradOutput):        
        self.gradInput = np.zeros_like(input)
        np.matmul(gradOutput, self.W.astype(input.dtype), out=self.gradInput)
        return self.gradInput
    
    def accGradParameters(self, input, gradOutput):
        self.gradW = gradOutput.T @ input
        self.gradb = gradOutput.sum(axis=0)
        
        pass
    
    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)
        
    def getParameters(self):
        return [self.W, self.b]
    
    def getGradParameters(self):
        return [self.gradW, self.gradb]
    
    def __repr__(self):
        s = self.W.shape
        q = 'Linear %d -> %d' %(s[1],s[0])
        return q

# SoftMax
class SoftMax(Module):
    def __init__(self):
         super(SoftMax, self).__init__()
    
    def updateOutput(self, input_):
        # start with normalization for numerical stability
        self.output = np.subtract(input_, input_.max(axis=1, keepdims=True))
        
        self.output = np.exp(self.output)
        self.output = self.output / np.sum(self.output, axis=1, keepdims=True)
        
        return self.output
    
    def updateGradInput(self, input_, gradOutput):
        local_repr_1 = np.einsum('bi,bj->bij', self.output, self.output)
        local_repr_2 = np.einsum(
            'bi,ij->bij',
            self.output,
            np.eye(input_.shape[1], input_.shape[1])
        )
        local_repr_3 = local_repr_2 - local_repr_1
        
        self.gradInput = np.einsum('bij,bi->bj', local_repr_3, gradOutput)
        return self.gradInput
    
    def __repr__(self):
        return "SoftMax"

# LogSoftMax
class LogSoftMax(Module):
    def __init__(self):
         super(LogSoftMax, self).__init__()
    
    def updateOutput(self, input):
        # start with normalization for numerical stability
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        
        self.output = self.output - np.log(np.sum(np.exp(self.output), axis=1, keepdims=True))
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        self.gradInput = np.zeros(input.shape)
        for i in range(input.shape[0]):
            self.gradInput[i] = gradOutput[i] @ np.subtract(np.eye(input.shape[1]),np.exp(self.output)[i])
        return self.gradInput
    
    def __repr__(self):
        return "LogSoftMax"

# Batch normalization
class BatchNormalization(Module):
    EPS = 1e-3
    def __init__(self, alpha = 0.):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = 0
        self.moving_variance = 0
        
    def updateOutput(self, input):
        self.output = np.zeros_like(input)
        if self.training == True:
            self.batch_mean = input.mean(axis=0)
            
            self.batch_var = input.var(axis=0)
            self.moving_mean = self.moving_mean * self.alpha + self.batch_mean * (1 - self.alpha)
            self.moving_variance = self.moving_variance * self.alpha + self.batch_var * (1 - self.alpha)
            self.output = np.subtract(input, self.batch_mean) / np.sqrt(self.batch_var + self.EPS)
        else:
            self.output = np.subtract(input, self.moving_mean) / np.sqrt(self.moving_variance + self.EPS)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        self.gradInput = np.zeros_like(input)
        normalized = np.zeros_like(input)
        normalized = np.subtract(input,self.batch_mean) / (np.sqrt(np.add(self.batch_var,self.EPS)))
        np.multiply(np.divide(1,np.multiply(np.sqrt(np.add(self.batch_var,self.EPS)), input.shape[0])),np.subtract(np.subtract(np.multiply(input.shape[0],gradOutput),gradOutput.sum(axis=0)),np.multiply(normalized,np.sum(np.multiply(gradOutput,normalized), axis=0))), out = self.gradInput)
        return self.gradInput
    
    def __repr__(self):
        return "BatchNormalization"

# Channelwise scaling
class ChannelwiseScaling(Module):
    """
       Implements linear transform of input y = \gamma * x + \beta
       where \gamma, \beta - learnable vectors of length x.shape[-1]
    """
    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()

        stdv = 1./np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)
        
        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def updateOutput(self, input):
        self.output = input * self.gamma + self.beta
        return self.output
        
    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.gamma
        return self.gradInput
    
    def accGradParameters(self, input, gradOutput):
        self.gradBeta = np.sum(gradOutput, axis=0)
        self.gradGamma = np.sum(gradOutput*input, axis=0)
    
    def zeroGradParameters(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)
        
    def getParameters(self):
        return [self.gamma, self.beta]
    
    def getGradParameters(self):
        return [self.gradGamma, self.gradBeta]
    
    def __repr__(self):
        return "ChannelwiseScaling"

# Dropout
class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        
        self.p = p
        self.mask = None
        
    def updateOutput(self, input):
        if not self.training:
            self.output = input
        else:
            self.mask = np.random.binomial(1, 1. - self.p, input.shape)
            self.output = np.multiply(input, self.mask)
            self.output /= 1. - self.p
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        if not self.training:
            self.gradInput = gradOutput
        else:
            self.gradInput = np.multiply(gradOutput, self.mask)
            self.gradInput /= (1. - self.p)
        return self.gradInput
        
    def __repr__(self):
        return "Dropout"

# ReLU
class ReLU(Module):
    def __init__(self):
         super(ReLU, self).__init__()
    
    def updateOutput(self, input):
        self.output = np.maximum(input, 0)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput , input > 0)
        return self.gradInput
    
    def __repr__(self):
        return "ReLU"

# Leaky ReLU
class LeakyReLU(Module):
    def __init__(self, slope = 0.03):
        super(LeakyReLU, self).__init__()
            
        self.slope = slope
        
    def updateOutput(self, input):
        self.output = np.maximum(input, 0)
        self.output -= self.slope * np.maximum(-input, 0)
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        mask_positive, mask_negative = input >= 0, input < 0
        self.gradInput = gradOutput*(mask_positive + mask_negative * self.slope) 
        return self.gradInput
    
    def __repr__(self):
        return "LeakyReLU"

# ELU
class ELU(Module):
    def __init__(self, alpha = 1.0):
        super(ELU, self).__init__()
        
        self.alpha = alpha
        
    def updateOutput(self, input):
        self.output = input.copy()
        self.output[self.output < 0] = (np.exp(self.output[self.output < 0])-1)*self.alpha
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        mask_positive, mask_negative = input >= 0, input < 0
        self.gradInput = gradOutput * (mask_positive + np.exp(input)*mask_negative*self.alpha)
        return self.gradInput
    
    def __repr__(self):
        return "ELU"

# SoftPlus
class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()
    
    def updateOutput(self, input):
        self.output = np.log(1 + np.exp(input))
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.divide(gradOutput, (np.exp(-input)+1))
        return self.gradInput
    
    def __repr__(self):
        return "SoftPlus"

# Sigmoid
class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def updateOutput(self, input):
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(np.multiply(gradOutput, self.output), 1-self.output)
        return self.gradInput

    def __repr__(self):
        return "Sigmoid"

# Tanh
class Tanh(Module):
    def __init__(self):
        super().__init__()
    
    def updateOutput(self, input):
        self.output = np.tanh(input)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput, 1 / (np.cosh(input)**2))
        return self.gradInput

    def __repr__(self):
        return 'Tanh'

class Swish(Module):
    def __init__(self):
        super().__init__()
    
    def updateOutput(self, input):
        self.output = np.multiply(input, 1 / (1 + np.exp(-input)))
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput, self.output + np.multiply(1 / (1 + np.exp(-input)), (1 - self.output)))
        return self.gradInput

    def __repr__(self):
        return 'Swish'

    