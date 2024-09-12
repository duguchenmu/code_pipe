from abc import ABCMeta, abstractmethod
import omegaconf
from omegaconf import OmegaConf
from torch import nn
from copy import copy

class MetaModel(ABCMeta):
    def __prepare__(name, bases, **kwds):
        total_conf = OmegaConf.create()
        for base in bases:
            for key in ('base_default_conf', 'default_conf'):
                update = getattr(base, key, {})
                if isinstance(update, dict):
                    update = OmegaConf.create(update)
                total_conf = OmegaConf.merge(total_conf, update)
        return dict(base_default_conf=total_conf)


class BaseModel(nn.Module, metaclass=MetaModel):
    required_data_keys = []
    def __init__(self, conf):
        super().__init__()
        self._init(conf)

    def forward(self, data):
        def recursive_key_check(expected, given):
            for key in expected:
                assert key in given, f'Missing key {key} in data'
                if isinstance(expected, dict):
                    recursive_key_check(expected[key], given[key])

        recursive_key_check(self.required_data_keys, data)
        return self._forward(data)

    @abstractmethod
    def _init(self, conf):
        raise NotImplementedError

    @abstractmethod
    def _forward(self, data):
        raise NotImplementedError