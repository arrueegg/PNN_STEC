"""
Utility functions for model manipulation, including parameter freezing for transfer learning.
"""

import logging

def freeze_model_body(model, config, logger):
    """
    Freeze all model parameters except the output head layer(s).
    
    This is used for transfer learning / finetuning scenarios where we want to
    keep the learned representations in the body and only adapt the final prediction layer.
    
    The output head can be named:
    - 'output_layer' for most models (MLP, BNN, ResNet, Attention models)
    - 'fusion' for Branch models (Branch_*, Branch3Way_*)
    
    For FactorizedSTECModelWrapper, supports selective freezing:
    - freeze_vtec_net: Freeze VTEC field network (keep ionospheric climatology)
    - freeze_geom_net: Freeze geometry/MF network (keep geometric model)
    
    Args:
        model: The PyTorch model to freeze
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        tuple: (frozen_params_count, trainable_params_count)
    """
    # Check if this is a factorized model with selective freezing
    from model.model import FactorizedSTECModelWrapper
    if isinstance(model, FactorizedSTECModelWrapper):
        return freeze_factorized_model(model, config, logger)
    
    # Standard freezing for other models
    freeze_body = config.get("finetune", {}).get("freeze_body", False)
    
    if not freeze_body:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"All model parameters will be trained: {total_params:,} total")
        return 0, total_params
    
    # Determine output head name based on model architecture
    output_head_names = ["output_layer", "fusion"]
    
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        # Check if this parameter belongs to the output head
        is_output_head = any(head_name in name for head_name in output_head_names)
        
        if not is_output_head:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            trainable_params += param.numel()
    
    logger.info(
        f"Froze model body parameters: {frozen_params:,} frozen, "
        f"{trainable_params:,} trainable (output head only)"
    )
    
    # Log which output head was found (for debugging)
    found_heads = [name for name in output_head_names if any(name in p for p, _ in model.named_parameters())]
    if found_heads:
        logger.info(f"Output head layer(s): {', '.join(found_heads)}")
    else:
        logger.warning("No recognized output head found! All parameters may be frozen.")
    
    return frozen_params, trainable_params


def freeze_factorized_model(model, config, logger):
    """
    Freeze subnets of a FactorizedSTECModelWrapper based on configuration.
    
    This allows selective freezing of:
    - VTEC field network: freeze_vtec_net=True (keep ionospheric field fixed)
    - Geometry/MF network: freeze_geom_net=True (keep geometric model fixed)
    
    Recommended fine-tuning strategy:
    - Daily fine-tuning: freeze_geom_net=True, adapt VTEC to daily ionospheric conditions
      → MF is deterministic physics (elevation → slant path), doesn't change day-to-day
      → VTEC changes with space weather, local time, ionospheric dynamics
    - Station calibration: freeze_vtec_net=True, adapt MF for new receivers/stations
      → When deploying to new stations, keep learned VTEC field, calibrate geometry
    
    Args:
        model: FactorizedSTECModelWrapper instance
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        tuple: (frozen_params_count, trainable_params_count)
    """
    freeze_vtec = config.get("finetune", {}).get("freeze_vtec_net", False)
    freeze_geom = config.get("finetune", {}).get("freeze_geom_net", False)
    
    frozen_params = 0
    trainable_params = 0
    
    # Access the wrapped factorized model
    factorized_model = model.model
    
    # Freeze VTEC network if requested
    if freeze_vtec:
        for param in factorized_model.vtec_net.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        logger.info("Froze VTECFieldNet (ionospheric field climatology preserved)")
    else:
        for param in factorized_model.vtec_net.parameters():
            trainable_params += param.numel()
    
    # Freeze Geometry/MF network if requested
    if freeze_geom:
        for param in factorized_model.geom_net.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        logger.info("Froze GeomNet (mapping factor model preserved)")
    else:
        for param in factorized_model.geom_net.parameters():
            trainable_params += param.numel()
    
    # Summary
    if freeze_vtec and freeze_geom:
        logger.warning("Both VTECNet and GeomNet are frozen! No parameters will be trained.")
    elif not freeze_vtec and not freeze_geom:
        logger.info(f"All FactorizedSTEC parameters will be trained: {trainable_params:,} total")
    else:
        logger.info(
            f"FactorizedSTEC selective freezing: {frozen_params:,} frozen, "
            f"{trainable_params:,} trainable"
        )
    
    return frozen_params, trainable_params


def get_trainable_parameters(model):
    """
    Get an iterator over only the trainable (requires_grad=True) parameters.
    
    This is useful when passing parameters to optimizers, especially when
    some parameters have been frozen via freeze_model_body().
    
    Args:
        model: PyTorch model
        
    Returns:
        filter iterator over trainable parameters
    """
    return filter(lambda p: p.requires_grad, model.parameters())


def count_parameters(model, only_trainable=False):
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        only_trainable: If True, only count trainable parameters
        
    Returns:
        int: Number of parameters
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())
