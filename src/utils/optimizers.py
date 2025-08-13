import torch
import torch.optim as optim

def get_optimizer(config, model_parameters):
    optimizer_type = config['training']['optimizer']
    # Use correct learning rate based on mode
    if config['mode'] == 'pretrain':
        lr = config['pretrain']["learning_rate"] 
    else:
        lr = config['finetune']["learning_rate"]
    weight_decay = config['training']["weight_decay"]

    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model_parameters, lr=lr, weight_decay=weight_decay)  # Stochastic Gradient Descent
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)  # Adam optimizer
    elif optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)  # Adam with weight decay fix
    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(model_parameters, lr=lr, weight_decay=weight_decay)  # RMSprop optimizer
    elif optimizer_type == 'Adagrad':
        optimizer = optim.Adagrad(model_parameters, lr=lr, weight_decay=weight_decay)  # Adagrad optimizer
    elif optimizer_type == 'Adadelta':
        optimizer = optim.Adadelta(model_parameters, lr=lr, weight_decay=weight_decay)  # Adadelta optimizer
    else:
        raise Exception(f'Unknown optimizer {optimizer_type}')

    return optimizer

def get_scheduler(config, optimizer):
    if config['mode'] == 'finetune':
        scheduler_type = config['training']['scheduler']
    elif config['mode'] == 'pretrain':
        scheduler_type = config['pretrain']['scheduler']

    if scheduler_type == 'none' or scheduler_type is None:
        return None
    elif scheduler_type == 'StepLR':
        step_size = 1000
        gamma = 0.1
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'ExponentialLR':
        gamma = 0.95
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_type == 'CosineAnnealingLR':
        T_max = config['pretrain']['epochs'] # Total number of epochs for the cosine annealing
        eta_min = config['pretrain']['learning_rate'] * 0.001 # Minimum learning rate
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    else:
        raise Exception(f'Unknown scheduler {scheduler_type}')

    return scheduler