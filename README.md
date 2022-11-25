# Differentiable Temporal Logic for Temporal Action Segmentation

This repository contains the software used in our NeurIPS 2022 Paper _Don't Pour Cereal into Coffee: Differentiable Temporal Logic for Temporal Action Segmentation_.

## Setup

The codebase is built on `PyTorch 1.10` and `Ubuntu 20.04`.
In principle it should work on `PyTorch>=1.0`.

1. Clone this repository.
2. Install dependencies listed in `requirements.txt`. You may need to install [PyTorch](https://pytorch.org/get-started/locally/) if you haven't done so.
3. Download dataset tarballs [here](https://drive.google.com/drive/folders/1j19xtl6HjqtSr0TyLfOTp3R8N5wuU1I-?usp=share_link). Decompress them into `dataset/`.

## Basic Usage

```
python main.py -f ${CONFIG_FILE} [-c <KEY1 VAL1> [KEY2 VAL2]...]
```

To run a predefined experiment, replace `${CONFIG_FILE}` with one of the `yml` files
in `config/`.
For example, `python main.py -f config/gru_50salads.yml`.
Key-value pairs following `-c` can be used to override configs.

If you feel like having fun and want to run your own experiments, please check `config/defaults.py` for config keys.
You will need to write a [yacs](https://github.com/rbgirshick/yacs) config file like those `config/*.yml`.

## Use DTL in Your Projects

It is straightforward to incorporate DTL into your own projects. 
We suggest the following workflow, 
which would introduce a small modification to your existing codebase.

1. Build constraints and compile them into a formula. Please check out `notebooks/rule_extractor.ipynb` for how it was done in our work. The process could be different in your specific case.
2. In your training loop, add components for DTL. For example:
```python
from lib.tl import parse

# Existing codebase for model and optimizer initialization...
model = Model()
# omitted ...

# Get ready the logic evaluator
## Create a formula
formula_str = '(F pour_coffee & F boil_water) -> (~pour_coffee W boil_water)'
## Parse the formula to get an evaluator
evaluator = parse(formula_str)
## An ap_map tells the evaluator how to map the name of a proposition to an 
## index, through which it can access the logits of that proposition.
list_of_props = ['pour_coffee', 'boil_water']
ap_map = lambda x: list_of_props.index(x)
## rho is the parameter used by _smin() and _smax() for smooth min() and max()
## Check lib.tl.evaluator._smin and lib.tl.evaluator._smax for details
rho = 1
logic_weight = 0.1

# The training loop
while epoch in range(max_epoch):
    for data in dataloader:
        X, Y = data
        # Y_ should be shaped (batch, num_propositions, time_steps)
        Y_ = model(X)
        
        # Compute the task loss, e.g., nll_loss
        task_loss = task_loss_function(Y, Y_, *args, **kwargs)

        # Compute the logic loss
        tl_score = evaluator(Y_, ap_map=ap_map, rho=rho)
        logic_loss = torch.log(1+torch.exp(-tl_score))
        
        # Combine loss functions
        (task_loss + logic_weight * logic_loss).backward()

        # continue the training loop ...
```


## Credits

The formula parsing code is inspired by [mvcisback/py-metric-temporal-logic](https://github.com/mvcisback/py-metric-temporal-logic).

Some data handling and evaluation code are adapted from [yiskw713/asrf](https://github.com/yiskw713/asrf) and [ptirupat/MLAD](https://github.com/ptirupat/MLAD/).

We would like to thank the authors for their contribution.

## About

Please visit our [project page](https://diff-tl.github.io/) for more details.

Feel free to [contact me](mailto://ziwei-xu@comp.nus.edu.sg) should you have any questions.

If you find this repository useful in your research, please consider citing our paper:

```bibtex
@inproceedings{xu2022difftl,
      title={Don't Pour Cereal into Coffee: Differentiable Temporal Logic for Temporal Action Segmentation}, 
      author={Ziwei Xu and Yogesh S Rawat and Yongkang Wong and Mohan S Kankanhalli and Mubarak Shah},
      booktitle={{NeurIPS}},
      year={2022}
}
```

## License

MIT License.
