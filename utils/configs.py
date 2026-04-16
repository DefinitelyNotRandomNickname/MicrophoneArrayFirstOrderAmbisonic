import yaml
import os
import re
from copy import deepcopy


def load_yaml(path, loader):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r") as f:
        return yaml.load(f, Loader=loader)


def deep_merge(a, b):
    """
    Recursively merge dict b into dict a
    """
    out = deepcopy(a)

    for k, v in b.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)

    return out


def load_and_merge_configs(data_cfg_path, model_cfg_path, train_cfg_path):
    """
    Loads 3 YAML configs and merges them into one dict
    """
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.')
    )

    data_cfg = load_yaml(data_cfg_path, loader)
    model_cfg = load_yaml(model_cfg_path, loader)
    train_cfg = load_yaml(train_cfg_path, loader)

    cfg = deep_merge(data_cfg, model_cfg)
    cfg = deep_merge(cfg, train_cfg)

    return cfg
