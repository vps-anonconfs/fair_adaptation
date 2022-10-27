from .expts import expts
from .dataset_defaults import dataset_defaults
from .model_defaults import model_defaults
from argparse import Namespace


def merge(cli_args, config_args):
    cli_dict, config_dict = vars(cli_args), vars(config_args)
    # overwrite cli if provided in config
    for k, v in config_dict.items():
        cli_dict[k] = v

    new_ns = Namespace(**cli_dict)
    return new_ns

def populate_defaults(config):
    """
    expts > dataset config > model config
    """
    assert config.name in expts, f"{config.name} not found in {list(expts.keys())}"
    populate_config(config, expts[config.name])

    assert hasattr(config, 'dataset') and config.dataset, f"Dataset not provided!!"
    assert config.dataset in dataset_defaults, f"{config.dataset} not in supported list {list(dataset_defaults.keys())}"
    populate_config(config, dataset_defaults[config.dataset])

    assert hasattr(config, 'model') and config.model, f"Model not provided!!"
    assert config.model in model_defaults, f"{config.model} not in supported list {list(model_defaults.keys())}"
    return populate_config(config, model_defaults[config.model])


def populate_config(config, template: dict, force_compatibility=False):
    """Populates missing (key, val) pairs in config with (key, val) in template.
    Example usage: populate config with defaults
    Args:
        - config: namespace
        - template: dict
        - force_compatibility: option to raise errors if config.key != template[key]
    """
    if template is None:
        return config

    d_config = vars(config)
    for key, val in template.items():
        if type(val) != dict:
            if key not in d_config or d_config[key] is None:
                d_config[key] = val
            elif d_config[key] != val and force_compatibility:
                raise ValueError(f"Argument {key} must be set to {val}")
        else:
            if key not in d_config or d_config[key] is None:
                d_config[key] = val
            else:
                for k, v in val.items():
                    if k not in d_config[key]:
                        d_config[key][k] = v
    return config
