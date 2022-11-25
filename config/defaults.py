"""Defaults used by yacs."""

import os

from .yacs import CfgNode as CN

_C = CN(new_allowed=True)

_C.seed = 0
_C.max_epoch = 1000
# Number of epochs to continue when test acc stagnants.
_C.patience = 250
_C.save_model_to = 'snapshots'

_C.root = 'data'
_C.config_path = 'config/defaults.py'

# Must be a name to a defined dataset class.
# For example, lib.dataset.asdataset.ActionSegmentationDataset is defined in $ROOT/lib/dataset/asdataset.py.
_C.ds_name = 'lib.dataset.asdataset.ActionSegmentationDataset'
# A dictionary passed to the constructor of the model class.
# Check $ROOT/lib/dataset/asdataset.py and see what keyword arguments are supported by the model class.
_C.ds_params = CN(new_allowed=True)

# Must be a name to a defined model class.
# For example, lib.model.gru.GRU is defined in $ROOT/lib/model/gru.py.
_C.model_name = 'lib.model.gru.GRU'
# A dictionary passed to the constructor of the model class.
# Check $ROOT/lib/model/gru.py and see what keyword arguments are supported by the model class.
_C.model_params = CN(new_allowed=True)

# Must be a name to a defined callable object (e.g. a function).
_C.loss_func = 'lib.model.commons.loss_func_gru'

# A dictionary of weights for loss terms.
_C.loss_weights = CN(new_allowed=True)

# A named exported by torch.optim
_C.opt = 'Adam'
_C.opt_params = CN(new_allowed=True)
_C.opt_params.kwargs = CN(new_allowed=True)
_C.opt_params.kwargs.lr = 1e-4
_C.opt_params.kwargs.weight_decay = 1e-5

# A named exported by torch.optim.lr_scheduler
_C.lrsch = 'MultiStepLR'
_C.lrsch_params = CN(new_allowed=True)
_C.lrsch_params.args = []
_C.lrsch_params.kwargs = CN(new_allowed=True)
_C.gclipv = 10

_C.batch_size = 32
_C.forward_batch_size = 32
_C.test_batch_size = 32

# The name used to determine performance
_C.acc_name = 'acc'
_C.num_workers = min(os.cpu_count(), 4)

_C.plot_cm = False
_C.amp = 0

_C.val_every = 1
_C.debug_runs = 5

# Whether to enable progress bar.
_C.pbar = False
_C.log_dir = None
_C.debug_mode = False
_C.summary_to = None
# UUID of the model. Will be generated automatically if unspecified.
_C.uuid = None
# Flag for cuda. Will be automatically determined if unspecified.
_C.cuda = False
# Either 'cuda' or 'cpu'. Will be automatically determined if unspecified.
_C.device = 'cuda'
# List of cuda devices, functional only when parallel flag is set.
_C.device_list = [0]
# Flag for parallelization/
_C.parallel = False
_C.start_epoch = 0
_C.max_epoch = 100

# Disable training. Model will only be tested.
_C.test_only = False

# Load model, optimizer, lr_scheduler state etc from path.
_C.from_model = None
# Load model state from path. This will invalidate --finetune flag.
_C.load_weight_from = None
# List of names to be ignored when loading weights.
_C.ignored_names = []
# List of names to filter out names when loading weights.
_C.ignored_filter_names = []
# List of names to be frozen after weights are loaded.
_C.freeze_names = []
# List of names to filter the weights to be frozen.
_C.freeze_filter_names = []

# extra payloads
_C.extra_kwargs = CN(new_allowed=True)

_C.comment = ''

def get_cfg_defaults():
    return _C.clone()
