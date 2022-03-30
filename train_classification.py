import numpy as np
import os
import json


def mean_weights_grads_ratio_norm(model):
    weight_norms = []
    grad_norms = []
    for module_weights, module_grads in zip(model.getParameters(), model.getGradParameters()):
        for parameter, parameter_grad in zip(module_weights, module_grads):
            weight_norms.append(np.linalg.norm(parameter))
            grad_norms.append(np.linalg.norm(parameter_grad))

    return np.mean(np.array(grad_norms) / np.array(weight_norms))

def train(
    model, criterion, optimizer, optimizer_config, train_dataloader,
    val_dataloader=None, test_dataloader=None, epoch_num=10, reg_coef=None,
    optimizer_state=None, logging=False, model_name='model', regularization=None,
    log_dir='/home/ivainn/Alex/DL_homeworks/logs', log_name='log.json', epoch_lrs=None
    ):

    if optimizer_state is None:
        optimizer_state = {}
    train_loss_history = []
    train_accuracy_history = []
    train_grad_norm_history = []
    val_loss_history = []
    val_accuracy_history = []
    test_loss = 0
    test_accuracy = 0

    for epoch in range(epoch_num):
        # Training
        if epoch_lrs is not None:
            optimizer_state['learning_rate'] = epoch_lrs[epoch]

        model.train()
        for x_batch, y_batch in train_dataloader:
            
            model.zeroGradParameters()

            # Forward
            predictions = model.forward(x_batch)
            loss = criterion.forward(predictions, y_batch)

            # Backward
            dp = criterion.backward(predictions, y_batch)
            model.backward(x_batch, dp)

            # Update weights
            optimizer(
                model.getParameters(), 
                model.getGradParameters(), 
                optimizer_config,
                optimizer_state,
                regularization,
                reg_coef
                )  

            # Update history
            accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_batch, axis=1))/y_batch.shape[0]
            train_accuracy_history.append(accuracy)
            train_loss_history.append(loss)
            train_grad_norm_history.append(optimizer_config['learning_rate'] * np.mean(mean_weights_grads_ratio_norm(model)))
        
        # Evaluation
        if val_dataloader is None:
            continue

        model.evaluate()
        samples_num = 0
        loss = 0
        correct_predictions = 0
        batches_num = 0
        for x_batch, y_batch in val_dataloader:
            samples_num += x_batch.shape[0]
            batches_num += 1

            predictions = model.forward(x_batch)
            loss += np.sum(criterion.forward(predictions, y_batch))
            correct_predictions += np.sum(np.argmax(predictions, axis=1) == np.argmax(y_batch, axis=1))
        val_loss_history.append(loss / batches_num)
        val_accuracy_history.append(correct_predictions / samples_num)

    if test_dataloader is not None:
        model.evaluate()
        samples_num = 0
        loss = 0
        correct_predictions = 0
        for x_batch, y_batch in test_dataloader:
            samples_num += x_batch.shape[0]
            predictions = model.forward(x_batch)
            loss += np.sum(criterion.forward(predictions, y_batch))
            correct_predictions += np.sum(np.argmax(predictions, axis=1) == np.argmax(y_batch, axis=1))
        
        test_loss = loss / samples_num
        test_accuracy = correct_predictions / samples_num
    
    data_to_log = {
                'model_name': model_name,
                'train_loss_history': train_loss_history,
                'train_accuracy_history': train_accuracy_history,
                'train_grad_norm_history': train_grad_norm_history,
                'val_loss_history': val_loss_history,
                'val_accuracy_history': val_accuracy_history,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            }
    
    if logging:
        with open(os.path.join(log_dir, log_name), 'w') as f:
            json.dump(data_to_log, f)
    
    return data_to_log
    
