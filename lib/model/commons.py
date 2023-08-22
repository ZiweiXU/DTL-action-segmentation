import torch
from torch.nn import functional as F


def loss_func(y=None, ytl=None, Y=None, my=None, mY=None, **kwargs):
    default_lsw = dict.fromkeys(
        ['ce', 'mce', 'smt', 'lg'], 1.0)
    loss_weights = kwargs.get('loss_weights', default_lsw)
    mask = kwargs.get('mask', None)

    losses = {}

    if loss_weights['ce'] > 0:
        y_mask = y[mask==1,...]
        Y_mask = Y[mask==1].squeeze(0).long()
        losses['ce'] = F.cross_entropy(y_mask, Y_mask, reduction='mean')

    if my is not None and loss_weights['mce'] > 0:
        losses['mce'] = F.cross_entropy(my, mY, reduction='mean')
    
    if loss_weights['smt'] > 0:
        # https://github.com/yabufarha/ms-tcn/blob/master/model.py
        losses['smt'] = 0
        _mse = torch.nn.MSELoss(reduction='none')
        y_ = torch.log_softmax(y, -1)
        losses['smt'] += torch.mean(
                torch.clamp(_mse(F.log_softmax(y_[:, 1:, :], dim=1), F.log_softmax(y_.detach()[:, :-1, :], dim=1)), min=0, max=16)
                *
                mask[:, 1:, None].to(y.device)
            )

    if loss_weights['lg'] > 0 and kwargs.get('rules_from', None) is not None and ytl is not None:
        losses['lg'] = 0
        rule_eva = kwargs['rules_from'].rule_eva
        ap_map = kwargs['rules_from'].ap_map

        y_ = torch.softmax(ytl, -1)
        if my is not None: 
            my_ = torch.softmax(my, -1)
            y_ = torch.cat([y_, my_.unsqueeze(1).repeat(1, y_.shape[1], 1)], 2)
        for i in range(y_.shape[0]):
            lg_eval_result = rule_eva(y_[i,(mask[i]==1)].transpose(0,1), ap_map=ap_map)
            losses['lg'] += (1 - lg_eval_result)**2
        losses['lg'] /= y_.shape[0]

    return losses, default_lsw


def loss_func_ref(y: torch.Tensor, Y: torch.Tensor, lg_eva=None, **kwargs):
    default_lsw = dict.fromkeys(
        ['lg', 'l2', 'bce'], 0.0)
    loss_weights = kwargs.get('loss_weights', default_lsw)
    losses = {}

    if loss_weights['lg'] > 0:
        assert (lg_eva is not None)
        losses['lg'] = 0
        for i in range(y.shape[0]):
            y_ = F.softmax(y[i], dim=0)
            losses['lg'] += (1 - lg_eva(y_))**2
        losses['lg'] /= y.shape[0]

    if loss_weights['l2'] > 0:
        losses['l2'] = ((y - Y)**2).sum(2).mean()
    
    if loss_weights['bce'] > 0:
        y_ = F.sigmoid(y)
        losses['bce'] = F.binary_cross_entropy(y_, Y, reduction='none').sum(2).mean()

    return losses, loss_weights


def _tl_loss(eva):
    return torch.log(1+torch.exp(-eva))


def loss_func_gru(y: torch.Tensor, Y: torch.Tensor, **kwargs):
    """
    Args:
        - y : Tensor of shape BxCxT
        - Y : Tensor of shape BxT
    """
    default_lsw = dict.fromkeys(
        ['ce', 'mce', 'lg'], 0.0)
    loss_weights = kwargs.get('loss_weights', default_lsw)
    
    losses = {}

    if loss_weights.get('ce', 0) > 0:
        losses['ce'] = F.cross_entropy(y[0].transpose(0,1), Y[0])
    
    if loss_weights.get('mce', 0) > 0 and kwargs['ym'] is not None:
        Ym, ym= kwargs['Ym'], kwargs['ym']
        Cm = ym.shape[1]
        losses['mce'] = F.cross_entropy(ym.view(-1, Cm), Ym.view(-1), ignore_index=-100)
    
    if loss_weights.get('lg', 0) > 0:
        assert 'lg_eva' in kwargs
        tl_eva = kwargs['lg_eva']
        y_ = y[0]
        if kwargs['ym'] is not None:
            ym = kwargs['ym']
            ym_ = ym.transpose(0,1).expand([-1, y_.shape[1]])
            y_ = torch.cat([y_, ym_], dim=0)
        else:
            pass
        losses['lg'] = _tl_loss(tl_eva(y_))
    
    assert len(losses) > 0

    return losses, loss_weights


def loss_func_mstcn(y: torch.Tensor, Y: torch.Tensor, **kwargs):
    """
    Args:
        - y : Tensor of shape SxBxCxT
        - Y : Tensor of shape BxT
    """
    default_lsw = dict.fromkeys(
        ['ce', 'sm', 'lg'], 0.0)
    loss_weights = kwargs.get('loss_weights', default_lsw)
    
    losses = {}

    S, B, C, T = y.shape

    if loss_weights.get('ce', 0) > 0:
        losses['ce'] = 0
        for p in y:
            losses['ce'] += F.cross_entropy(p.transpose(1,2).contiguous().view(-1, C), Y.view(-1), ignore_index=-100)
    
    if loss_weights.get('sm', 0) > 0:
        mask = kwargs['mask']
        losses['sm'] = 0
        for p in y:
            losses['sm'] += torch.mean(
                torch.clamp(
                    F.mse_loss(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1), reduction='none'), 
                min=0, max=16)
                *mask[:, :, 1:]
            )
    
    if loss_weights.get('lg', 0) > 0:
        assert 'lg_eva' in kwargs
        tl_eva = kwargs['lg_eva']
        losses['lg'] = 0
        y_ = y[-1][0]
        losses['lg'] += _tl_loss(tl_eva(y_))
    
    assert len(losses) > 0

    return losses, loss_weights
