import numpy as np

# Most part of this code was written as a homework on mipt-ml-base course: https://github.com/girafe-ai/ml-mipt (Neychev et al.)

def apply_regularization(variables, regularization, coef):
    """
        Get additional gradients due to regularization ('l2', 'l1')
        `variables` - list of lists of variables (one list per layer)
    """
    if regularization.lower() == 'l2':
        for current_layer_vars in variables: 
            for current_var in current_layer_vars:
                np.subtract(current_var, coef * 2 * current_var, current_var)
                
    elif regularization.lower() == 'l1':
        for current_layer_vars in variables: 
            for current_var in current_layer_vars:
                mask = (current_var > 0).astype(current_var.dtype) - (current_var < 0).astype(current_var.dtype)
                np.subtract(current_var, coef * mask, current_var)



# SGD optimizer with momentum
def sgm_momentum(variables, gradients, config, state, regularization=None, coef=1e-2):
    """
        `variables` - list of lists of variables (one list per layer)
        `gradients` - list of lists of current gradients (same structure as for `variables`, one array for each var)
        `config` - dict with optimization parameters (`learning_rate` and `momentum`)
        `state` - dict with optimizator state (used to save accumulated gradients)
    """
    state.setdefault('accumulated_grads', {})
    
    var_index = 0 
    for current_layer_vars, current_layer_grads in zip(variables, gradients): 
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            
            old_grad = state['accumulated_grads'].setdefault(var_index, np.zeros_like(current_grad))
            
            np.add(config['momentum'] * old_grad, config['learning_rate'] * current_grad, out=old_grad)            
            current_var -= old_grad
            var_index += 1 

    if regularization is not None:
        apply_regularization(variables, regularization, coef)    

# Adam optimizer (https://arxiv.org/pdf/1412.6980.pdf)
def adam_optimizer(variables, gradients, config, state, regularization=None, coef=1e-2):  
    """
        `variables` - list of lists of variables (one list per layer)
        `gradients` - list of lists of current gradients (same structure as for `variables`, one array for each var)
        `config` - dict with optimization parameters (`learning_rate`, `beta1`, `beta2`, `epsilon`)
        `state` - dict with optimizator state (used to save 1st and 2nd moment for vars)
          
        Formulas for optimizer:
          
        Current step learning rate: $$\text{lr}_t = \text{learning_rate} * \frac{\sqrt{1-\beta_2^t}} {1-\beta_1^t}$$
        First moment of var: $$\mu_t = \beta_1 * \mu_{t-1} + (1 - \beta_1)*g$$ 
        Second moment of var: $$v_t = \beta_2 * v_{t-1} + (1 - \beta_2)*g*g$$
        New values of var: $$\text{variable} = \text{variable} - \text{lr}_t * \frac{m_t}{\sqrt{v_t} + \epsilon}$$
    """
    state.setdefault('m', {})  # first moment vars
    state.setdefault('v', {})  # second moment vars
    state.setdefault('t', 0)   # timestamp
    state['t'] += 1
    for k in ['learning_rate', 'beta1', 'beta2', 'epsilon']:
        assert k in config, config.keys()
    
    var_index = 0 
    lr_t = config['learning_rate'] * np.sqrt(1 - config['beta2']**state['t']) / (1 - config['beta1']**state['t'])
    for current_layer_vars, current_layer_grads in zip(variables, gradients): 
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            var_first_moment = state['m'].setdefault(var_index, np.zeros_like(current_grad))
            var_second_moment = state['v'].setdefault(var_index, np.zeros_like(current_grad))
            np.add(config['beta1'] * var_first_moment, (1-config['beta1']) * current_grad , out = var_first_moment)
            np.add(config['beta2'] * var_second_moment, (1-config['beta2']) * current_grad * current_grad, out = var_second_moment)
            current_var -= lr_t * var_first_moment / np.sqrt((var_second_moment) + config['epsilon'])
            
            assert var_first_moment is state['m'].get(var_index)
            assert var_second_moment is state['v'].get(var_index)
            var_index += 1
    
    if regularization is not None:
        apply_regularization(variables, regularization, coef)