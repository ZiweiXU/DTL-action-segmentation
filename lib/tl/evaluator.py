import os
from functools import singledispatch

import torch

from . import syntax as ast


def _ap_to_index(ap_name):
    """For now only allow [a-z][0-9]+ format."""
    return int(ap_name[1:])


def _agg_result(x, res):
    return torch.ones_like(x) * res


def _mask_ninf(x, mask):
    """
    Replace elements in tensor x based on mask.
    If mask[i] == 0, then x[i] will be set to -infinity.
    """
    x1 = (1 - mask).unsqueeze(1) * torch.inf
    x1[x1.isnan()] = 0
    return x - x1


def _mask_pinf(x, mask):
    """
    Replace elements in tensor x based on mask.
    If mask[i] == 0, then x[i] will be set to +infinity.
    """
    x1 = (1 - mask).unsqueeze(1) * torch.inf
    x1[x1.isnan()] = 0
    return x + x1


# https://mathoverflow.net/questions/35191/a-differentiable-approximation-to-the-minimum-function
def _smax(x, rho=100., dim=0, mask=None):
    if mask is not None and len(x.shape) > 2:
        x = _mask_ninf(x, mask)
    return torch.logsumexp(rho*x, dim=dim, keepdim=True) / rho


def _smin(x, rho=100., dim=0, mask=None):
    if mask is not None and len(x.shape) > 2:
        _mask_pinf(x, mask)
    return -torch.logsumexp(rho*(-x), dim=dim, keepdim=True) / rho


def _cummax(x, dim=1, rho=100.0, mask=None):
    if mask is not None and len(x.shape) > 2:
        x = _mask_ninf(x, mask)
    return torch.logcumsumexp(rho*x, dim=dim) / rho


def _cummin(x, dim=1, rho=100.0, mask=None):
    if mask is not None and len(x.shape) > 2:
        x = _mask_pinf(x, mask)
    return -torch.logcumsumexp(rho*(-x), dim=dim) / rho


# Evaluation cache: a dict of {str: Tensor}.
# This assumes that any formula is identified by how it is written.
_SYM_CACHE = {}
_CACHE_PLACEHOLDER = 'PLACEHOLDER'

def _register_cache(sym_name, value):
    global _SYM_CACHE
    _SYM_CACHE[sym_name] = value


def _clear_cache():
    global _SYM_CACHE
    _SYM_CACHE = {}


def _record_value(phi, value, level):
    if os.environ.get('TL_PRINT_TRACE', '0') == '1' or os.environ.get('TL_RECORD_TRACE', '0') == '1':
        phi.value = value
    else:
        return


def pointwise_sat(phi, ap_map=None, rho=100):
    _clear_cache()

    f = _eval_(phi, ap_map=ap_map, level=0, rho=rho)
    
    def _eval(x, t, *_, output_mask=None):
        x = f(x, t, output_mask=output_mask)
        if len(x.shape) == 2:
            return x[0][0]
        else:
            return x[:, 0, 0]

    return _eval


@singledispatch
def _eval_(phi, ap_map=None, **kwargs):
    """
    ---
    ### Args:
    - phi: the formula
    - ap_map: a callable that maps an atomic proposition to an index
    - rho: parameter used by _smax and _smin for smooth min() and max()

    ### Returns:
    A callable with signature f(x, t), where
    - x: is a numpy array of (N, L) where N is the number of proposition and L is the signal length
    - t: the start time of evaluation
    """
    raise NotImplementedError


@_eval_.register(ast.AtomicPred)
def _eval_ap(phi, *_, ap_map=None, level=0, rho=100):
    """This is where dispatch stops."""
    
    if phi.id not in _SYM_CACHE:
        _register_cache(phi.id, _CACHE_PLACEHOLDER)

        """This is where dispatch stops."""
        def _eval(x, t, output_mask=None):
            if ap_map is None:
                idx = _ap_to_index(phi.id)
            else:
                idx = ap_map(phi.id)

            if len(x.shape) == 2:
                value = x[idx:idx+1, t:]
            else:
                value = x[:, idx:idx+1, t:]
            _record_value(phi, value, level)
            _register_cache(phi.id, value)
            return value
        return _eval

    else:
        def _get_cache(x, t, output_mask=None):
            return _SYM_CACHE[str(phi)]
        return _get_cache


@_eval_.register(ast.Neg)
def _eval_neg(phi, ap_map=None, level=0, rho=100):
    
    if str(phi) not in _SYM_CACHE:
        _register_cache(str(phi), _CACHE_PLACEHOLDER)

        f = _eval_(phi.arg, ap_map=ap_map, level=level+1, rho=rho)
        def _eval(x, t, output_mask=None):
            x = f(x, t, output_mask=output_mask)
            value = -x
            _record_value(phi, value, level)
            _register_cache(str(phi), value)
            return value
        return _eval

    else:
        def _get_cache(x, t, output_mask=None):
            return _SYM_CACHE[str(phi)]
        return _get_cache


@_eval_.register(ast.And)
def _eval_and(phi, ap_map=None, level=0, rho=100):

    if str(phi) not in _SYM_CACHE:
        _register_cache(str(phi), _CACHE_PLACEHOLDER)

        fs = [_eval_(arg, ap_map=ap_map, level=level+1, rho=rho) for arg in phi.args]
        def _eval(x, t, output_mask=None):
            xs = []
            if len(x.shape) == 2:
                for f in fs:
                    xs.append(f(x,t,output_mask=output_mask))
                xs = [_smin(torch.cat(xs, dim=0), dim=0, rho=rho)]
            else:
                for f in fs:
                    xs.append(f(x,t,output_mask=output_mask))
                xs = [_smin(torch.cat(xs, dim=1), dim=1, rho=rho)]

            value = xs[0]
            _record_value(phi, value, level)
            _register_cache(str(phi), value)
            return value
        return _eval

    else:
        def _get_cache(x, t, output_mask=None):
            return _SYM_CACHE[str(phi)]
        return _get_cache


@_eval_.register(ast.Or)
def _eval_or(phi, ap_map=None, level=0, rho=100):

    if str(phi) not in _SYM_CACHE:
        _register_cache(str(phi), _CACHE_PLACEHOLDER)

        fs = [_eval_(arg, ap_map=ap_map, level=level+1, rho=rho) for arg in phi.args]
        def _eval(x, t, output_mask=None):
            xs = []
            if len(x.shape) == 2:
                for f in fs:
                    xs.append(f(x,t,output_mask=output_mask))
                xs = [_smax(torch.cat(xs, dim=0), dim=0, rho=rho)]
            else:
                for f in fs:
                    xs.append(f(x,t,output_mask=output_mask))
                xs = [_smax(torch.cat(xs, dim=1), dim=1, rho=rho)]

            value = xs[0]
            _record_value(phi, value, level)
            _register_cache(str(phi), value)

            return value
        return _eval

    else:
        def _get_cache(x, t, output_mask=None):
            return _SYM_CACHE[str(phi)]
        return _get_cache
    


@_eval_.register(ast.G)
def _eval_g(phi, ap_map=None, level=0, rho=100):

    if str(phi) not in _SYM_CACHE:
        _register_cache(str(phi), _CACHE_PLACEHOLDER)

        f = _eval_(phi.arg, ap_map=ap_map, level=level+1, rho=rho)
        def _eval(x, t, output_mask=None):
            x = f(x, t, output_mask=output_mask)
            if len(x.shape) == 2:
                value = _smin(x, dim=1, rho=rho)
            else:
                value = _smin(x, dim=2, rho=rho, mask=output_mask)
            value = _agg_result(x, value)
            _record_value(phi, value, level)
            _register_cache(str(phi), value)
            return value
        return _eval

    else:
        def _get_cache(x, t, output_mask=None):
            return _SYM_CACHE[str(phi)]
        return _get_cache


@_eval_.register(ast.WeakUntil)
def _eval_until(phi, ap_map=None, level=0, rho=100):

    if str(phi) not in _SYM_CACHE:
        _register_cache(str(phi), _CACHE_PLACEHOLDER)

        f1, f2 = _eval_(phi.arg1, ap_map=ap_map, level=level+1, rho=rho), _eval_(phi.arg2, ap_map=ap_map, level=level+1, rho=rho)
        def _eval(x, t, output_mask=None):
            x1, x2 = f1(x, t, output_mask=output_mask), f2(x, t, output_mask=output_mask)

            if len(x1.shape) == 2:
                x2_cummax = _cummax(x2, dim=1, rho=rho)
                value = _smin(_smax(torch.cat([x1, x2_cummax], dim=0), dim=0, rho=rho), dim=1, rho=rho)
            else:
                x2_cummax = _cummax(x2, dim=2, rho=rho, mask=output_mask)
                value = _smin(_smax(torch.cat([x1, x2_cummax], dim=1), dim=1, rho=rho), dim=2, rho=rho, mask=output_mask)
            value = _agg_result(x1, value)

            _record_value(phi, value, level)
            _register_cache(str(phi), value)

            return value
        return _eval

    else:
        def _get_cache(x, t, output_mask=None):
            return _SYM_CACHE[str(phi)]
        return _get_cache


@_eval_.register(ast.Since)
def _eval_since(phi, ap_map=None, level=0, rho=100):

    if str(phi) not in _SYM_CACHE:
        _register_cache(str(phi), _CACHE_PLACEHOLDER)

        f1, f2 = _eval_(phi.arg1, ap_map=ap_map, level=level+1, rho=rho), _eval_(phi.arg2, ap_map=ap_map, level=level+1, rho=rho)
        def _eval(x, t, output_mask=None):
            x1, x2 = f1(x, t, output_mask=output_mask), f2(x, t, output_mask=output_mask)

            if len(x1.shape) == 2:
                x2_cummax = _cummax(x2, dim=1, rho=rho)
                x2_mask = _cummin(x2_cummax * x2.sign(), dim=1, rho=rho)
                value = _smin(_smax(torch.cat([x1, x2_mask], dim=0), dim=0, rho=rho), dim=1, rho=rho)
                
            else:
                x2_cummax = _cummax(x2, dim=2, rho=rho, mask=output_mask)
                x2_mask = _cummin(x2_cummax * x2.sign(), dim=2, rho=rho, mask=output_mask)
                value = _smin(_smax(torch.cat([x1, x2_mask], dim=1), dim=1, rho=rho), dim=2, rho=rho, mask=output_mask)
            value = _agg_result(x1, value)

            _record_value(phi, value, level)
            _register_cache(str(phi), value)

            return value
        return _eval

    else:
        def _get_cache(x, t, output_mask=None):
            return _SYM_CACHE[str(phi)]
        return _get_cache


@_eval_.register(ast.Next)
def _eval_next(phi, ap_map=None, level=0, rho=100):

    if str(phi) not in _SYM_CACHE:
        _register_cache(str(phi), _CACHE_PLACEHOLDER)

        f = _eval_(phi.arg, ap_map=ap_map, level=level+1, rho=rho)
        def _eval(x, t, output_mask=None):
            x_next = f(x, t, output_mask=output_mask)[..., 1:]
            if len(x.shape) == 2:
                value = torch.cat([x_next, torch.ones(size=(x.shpae[0], 1))], dim=1)
            else:
                value = torch.cat([x_next, torch.ones(size=(x.shape[0], x.shape[1], 1))], dim=2)
            
            _record_value(phi, value, level)
            _register_cache(str(phi), value)
            
            return value
        return _eval

    else:
        def _get_cache(x, t, output_mask=None):
            return _SYM_CACHE[str(phi)]
        return _get_cache


@_eval_.register(type(ast.BOT))
def _eval_bot(phi, ap_map=None, level=0, rho=100):

    if str(phi) not in _SYM_CACHE:
        _register_cache(str(phi), _CACHE_PLACEHOLDER)
        def _eval(x, t, output_mask=None):
            if len(x.shape) == 2:
                value = torch.nn.Parameter(torch.zeros(1, x.shape[1]).to(x.device))
            else:
                value = torch.nn.Parameter(torch.zeros(1, x.shape[1]).to(x.device))
            
            _record_value(phi, value, level)
            _register_cache(str(phi), value)
            return value
        return _eval

    else:
        def _get_cache(x, t, output_mask=None):
            return _SYM_CACHE[str(phi)]
        return _get_cache
