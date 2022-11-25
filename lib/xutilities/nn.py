import logging

from torch import nn
from .utils import myself


class Module(nn.Module):
    """A subclass of torch.nn.Module with an additional state dict loader.
    """
    def __init__(self):
        super(Module, self).__init__()

    def forward(self, x):
        pass

    def load_my_state_dict(
        self, state_dict, 
        ignored_names=[], ignored_filter_names=[],
        freeze_names=[], freeze_filter_names=[]):
        # https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if (name in ignored_names) or (name not in own_state):
                logging.getLogger(myself()).warning(
                    f'Ignored: {name}.')
                continue

            filter_flag = False
            for ignore_filter in ignored_filter_names:
                if ignore_filter in name:
                    logging.getLogger(myself()).warning(
                        f'Filtered: {name}.')
                    filter_flag = True
                    break
            if filter_flag: continue

            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except RuntimeError as e:
                logging.error(f'Error loading {name}.')
                raise e


class SaveOutput:
    """Layer used as hooks."""
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []
