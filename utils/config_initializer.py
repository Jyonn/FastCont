import os

import yaml
from UniTok.classify import Classify

from utils.formatter import Formatter


def init_config(config_path, exp_path):
    config = yaml.safe_load(open(config_path))
    config = Classify(config)

    exp = yaml.safe_load(open(exp_path))
    exp = Classify(exp)

    exp.model = exp.model.upper()

    formatter = Formatter(
        model=exp.model,
        dataset=config.dataset,
        hidden_size=config.model_config.hidden_size,
        batch_size=exp.policy.batch_size,
    )

    if config.store.data_dir:
        config.store.data_dir = formatter(config.store.data_dir)
    config.store.save_dir = formatter(config.store.save_dir)

    config.store.ckpt_path = os.path.join(config.store.save_dir, exp.exp)
    config.store.log_path = os.path.join(config.store.ckpt_path, '{}.log'.format(exp.exp))

    os.makedirs(config.store.ckpt_path, exist_ok=True)

    return config, exp
