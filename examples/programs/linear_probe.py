"""cifar10-to-cifar100 baseline: keep the CIFAR-10 features, fit a ridge-regression head."""


def build(weights, data, info):
    features = toolkit.capture(weights, data['images'], ['avgpool'], info['num_classes'], reduce='pool')['avgpool']
    W, b = toolkit.ridge(features, data['labels'], info['num_classes'], lam=1e-3)
    weights['fc.weight'], weights['fc.bias'] = W, b
    return weights
