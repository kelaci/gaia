"""
Default configurations for GAIA.

This module provides sensible default configurations for all GAIA components,
making it easy to get started while allowing full customization.
"""

from typing import Dict, Any, List

# Default hierarchy configuration
DEFAULT_HIERARCHY_CONFIG: Dict[str, Any] = {
    "num_levels": 4,
    "temporal_compression": 2,  # Each level compresses time by this factor
    "base_resolution": 1,       # Base temporal resolution
    "level_sizes": [64, 128, 256, 512],  # Feature sizes at each level
    "communication_interval": 5,  # Steps between inter-level communication
    "max_levels": 8,             # Maximum number of levels allowed
    "min_levels": 2              # Minimum number of levels required
}

# Default plasticity configuration
DEFAULT_PLASTICITY_CONFIG: Dict[str, float] = {
    "learning_rate": 0.01,
    "ltp_coefficient": 1.0,      # Long-Term Potentiation strength
    "ltd_coefficient": 0.8,      # Long-Term Depression strength
    "decay_rate": 0.001,         # Weight decay rate
    "homeostatic_strength": 0.1, # Homeostatic regulation strength
    "bcm_theta": 1.0,            # BCM rule threshold
    "stdp_tau": 20.0             # STDP time constant
}

# Default ES (Evolutionary Strategy) configuration
DEFAULT_ES_CONFIG: Dict[str, Any] = {
    "population_size": 50,
    "sigma": 0.1,                # Mutation strength
    "learning_rate": 0.01,       # Mean update rate
    "elite_fraction": 0.2,      # Fraction of top performers to select
    "max_sigma": 1.0,            # Maximum mutation strength
    "min_sigma": 0.001,          # Minimum mutation strength
    "adaptation_rate": 0.01      # Sigma adaptation rate
}

# Default layer configurations
DEFAULT_LAYER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "reactive": {
        "activation": "relu",
        "init_type": "he",
        "use_bias": True,
        "bias_init": 0.0
    },
    "hebbian": {
        "plasticity_rule": "hebbian",
        "params": DEFAULT_PLASTICITY_CONFIG,
        "activity_trace_length": 100,
        "normalization": "l2"
    },
    "temporal": {
        "activation": "tanh",
        "time_window": 10,
        "recurrent_init": "he",
        "state_normalization": True
    }
}

# Default meta-learning configuration
DEFAULT_META_CONFIG: Dict[str, Any] = {
    "num_episodes": 100,
    "task_switch_frequency": 10,
    "performance_threshold": 0.8,
    "outer_optimizer": "adam",
    "adaptation_strategy": "uniform",
    "evaluation_interval": 5
}

# Default visualization configuration
DEFAULT_VIS_CONFIG: Dict[str, Any] = {
    "figsize": (12, 8),
    "dpi": 100,
    "colormap": "viridis",
    "linewidth": 2,
    "alpha": 0.7,
    "grid_alpha": 0.3
}

# Default logging configuration
DEFAULT_LOG_CONFIG: Dict[str, Any] = {
    "level": "INFO",
    "file_logging": False,
    "log_file": "gaia.log",
    "console_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}

def get_default_config(component: str) -> Dict[str, Any]:
    """
    Get default configuration for a specific component.

    Args:
        component: Component name ('hierarchy', 'plasticity', 'es', 'layers', 'meta', 'vis', 'log')

    Returns:
        Default configuration dictionary

    Raises:
        ValueError: If unknown component is specified
    """
    component = component.lower()

    if component == 'hierarchy':
        return DEFAULT_HIERARCHY_CONFIG.copy()
    elif component == 'plasticity':
        return DEFAULT_PLASTICITY_CONFIG.copy()
    elif component == 'es':
        return DEFAULT_ES_CONFIG.copy()
    elif component == 'layers':
        return DEFAULT_LAYER_CONFIGS.copy()
    elif component == 'meta':
        return DEFAULT_META_CONFIG.copy()
    elif component == 'vis':
        return DEFAULT_VIS_CONFIG.copy()
    elif component == 'log':
        return DEFAULT_LOG_CONFIG.copy()
    else:
        raise ValueError(f"Unknown component: {component}")

def validate_config(config: Dict[str, Any], config_type: str) -> bool:
    """
    Validate a configuration dictionary.

    Args:
        config: Configuration dictionary to validate
        config_type: Type of configuration to validate

    Returns:
        True if configuration is valid, False otherwise

    TODO:
        - Implement proper configuration validation
        - Add support for different configuration types
        - Consider schema-based validation
    """
    # Placeholder implementation
    return True

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration

    TODO:
        - Implement proper configuration merging
        - Add support for nested dictionaries
        - Consider different merge strategies
    """
    # Simple merge for now
    merged = base_config.copy()
    merged.update(override_config)
    return merged

def get_config_template(component: str) -> Dict[str, Any]:
    """
    Get a configuration template with descriptions.

    Args:
        component: Component name

    Returns:
        Configuration template with descriptions

    TODO:
        - Implement proper configuration templates
        - Add descriptions for each parameter
        - Consider different template formats
    """
    # Placeholder implementation
    return get_default_config(component)