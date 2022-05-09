# First task: Neural Net training pipeline with numpy
1. Implent fully connnected neural network pipeline using numpy, without using Deep Learning Frameworks

    a) Implement neural modules - **Sequential container**, **Linear**, **SoftMax**, **LogSoftMax**
  
    b) Implement activation modules - **ReLU**, **LeakyReLU**, **Sigmoid**, **Tanh**, **SoftPlus**, **ELU**, **Swish**

    This part is implemented in `modules/modules.py`

2. Implement **criterions**: **NegativeLogLikelyhood** loss, **MSE**, **Hinge** Loss
    
    This part is implemented in `modules/critetions.py`

3. Implement 2 optimizers - I've implemented **SGD Momentum** and **Adam**.

    This part is implemented in `modules/optimizers.py`

4. Implement Regularizations - **L1**, **L2** and **Dropout**
I've implemented Dropout as a neural module, L1 and L2 - as a additional argument in optimizers.

5. Conduct experiments showing the influence of various things on neural network training (regularization, butch size, hyperparameters, etc.)
    The experiments were carried out in `main_notebook.ipynb`
    
For pipeline I've implemented Dataset and Dataloader modules in `data.py`. The dataset used is [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

Main train loop I've implemented in `train_classification.py`.

