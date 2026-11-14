"""scratch-cifar10 baseline: keep random convs, set every BatchNorm from data, ridge head.

BatchNorm layers are calibrated in forward order so each one normalizes the
activations it actually receives; the head is then fit by least squares.
"""


def build(weights, data, info):
    k = info['num_classes']
    images = data['images'][:5000]
    bn_names = [name[:-len('.running_mean')] for name in weights if name.endswith('running_mean')]
    for bn in bn_names:
        conv = bn.replace('bn', 'conv') if 'shortcut' not in bn else bn[:-1] + '0'
        stats = toolkit.capture(weights, images, [conv], k, reduce='mean_var')[conv]
        weights[f'{bn}.running_mean'], weights[f'{bn}.running_var'] = stats['mean'], stats['var']
        weights[f'{bn}.weight'] = torch.ones_like(stats['mean'])
        weights[f'{bn}.bias'] = torch.zeros_like(stats['mean'])
    features = toolkit.capture(weights, data['images'], ['avgpool'], k, reduce='pool')['avgpool']
    weights['fc.weight'], weights['fc.bias'] = toolkit.ridge(features, data['labels'], k, lam=1e-2)
    return weights
