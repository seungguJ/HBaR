from .. import *
from .  import *

from ..model.lenet import *
from ..model.vgg import *
from ..model.resnet import *
from ..model.wideresnet import *

from torch.optim.lr_scheduler import _LRScheduler
def activations_extraction(model, data_loader, out_dim=10, hid_idx=-1,):

    out_activation = np.zeros([len(data_loader)*data_loader.batch_size, out_dim])
    out_label = np.zeros([len(data_loader)*data_loader.batch_size,])
    device = next(model.parameters()).device

    for batch_idx, (data, target) in enumerate(data_loader):
        
        if len(data)<data_loader.batch_size:
            break

        data = data.to(device)
        output, hiddens = model(data)
        
        begin = batch_idx*data_loader.batch_size
        end = (batch_idx+1)*data_loader.batch_size
        out_activation[begin:end] = hiddens[hid_idx].detach().cpu().numpy()
        out_label[begin:end] = target.detach().cpu().numpy()
        
    return {"activation":out_activation, "label":out_label}


def hsic_objective(hidden, h_target, h_data, sigma, k_type_y='gaussian'):


    hsic_hy_val = hsic_normalized_cca( hidden, h_target, sigma=sigma, k_type_y=k_type_y)
    hsic_hx_val = hsic_normalized_cca( hidden, h_data,   sigma=sigma)

    return hsic_hx_val, hsic_hy_val

def set_optimizer(config_dict, model, train_loader):
    """ bag of tricks set-ups"""
    config_dict['smooth'] = config_dict['smooth_eps'] > 0.0
    config_dict['mixup'] = config_dict['alpha'] > 0.0

    optimizer_init_lr = config_dict['warmup_lr'] if config_dict['warmup'] else config_dict['learning_rate']

    optimizer = torch.optim.Adam(model.parameters(), optimizer_init_lr) 
    scheduler = None
    if config_dict['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config_dict['epochs'] * len(train_loader), eta_min=4e-08)
    else:
        """Set the learning rate of each parameter group to the initial lr decayed
                by gamma once the number of epoch reaches one of the milestones
        """
        if config_dict['data_code'] == 'mnist':
            epoch_milestones = [65, 90]
        elif config_dict['data_code'] == 'cifar10':
            epoch_milestones = [65, 100, 130, 190, 220, 250, 280]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i * len(train_loader) for i in epoch_milestones], gamma=0.5)
        
    if config_dict['warmup']:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=config_dict['learning_rate']/config_dict['warmup_lr'], total_iter=config_dict['warmup_epochs'] * len(train_loader), after_scheduler=scheduler)
        
    return optimizer, scheduler


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()

def model_distribution(config_dict):
    if config_dict['data_code'] == 'cifar100':
        num_classes = 100
    else:
        num_classes = 10
    if config_dict['model'] == 'resnet18':
        model = ResNet18(**config_dict, num_classes=num_classes)
    elif config_dict['model'] == 'resnet50':
        model = ResNet50(**config_dict, num_classes=num_classes)
    elif config_dict['model'] == 'wideresnet-28-10':
        model = wideresnet(config_dict['model'], num_classes=num_classes, **config_dict)
    else:
        raise ValueError("Unknown model name or not support [{}]".format(config_dict['model']))

    return model

class CrossEntropyLossMaybeSmooth(nn.CrossEntropyLoss):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    def __init__(self, smooth_eps=0.0):
        super(CrossEntropyLossMaybeSmooth, self).__init__()
        self.smooth_eps = smooth_eps

    def forward(self, output, target, smooth=False):
        if not smooth:
            return F.cross_entropy(output, target)

        target = target.contiguous().view(-1)
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        smooth_one_hot = one_hot * (1 - self.smooth_eps) + (1 - one_hot) * self.smooth_eps / (n_class - 1)
        log_prb = F.log_softmax(output, dim=1)
        loss = -(smooth_one_hot * log_prb).sum(dim=1).mean()
        return loss


def mixup_data(x, y, alpha=1.0):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam, smooth):
    return lam * criterion(pred, y_a, smooth=smooth) + \
           (1 - lam) * criterion(pred, y_b, smooth=smooth)

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_iter: target learning rate is reached at total_iter, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_iter, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_iter = total_iter
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_iter:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_iter + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)
        

def trades_loss(config_dict,
                output,
                adv_output,
                y,
                beta=6.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    batch_size = output.size(0)
    # zero gradient
    # calculate robust loss
    # logits = model(x_natural)
    criterion = CrossEntropyLossMaybeSmooth(smooth_eps=config_dict['smooth_eps'])
    loss_natural = criterion(output, y, smooth=config_dict['smooth'])
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_output, dim=1),
                                                    torch.clamp(F.softmax(output, dim=1), min=1e-8))
    loss = loss_natural + beta * loss_robust
    return loss

def mart_loss(config_dict,
              output,
              adv_output,
              y,
              beta=6.0):
    kl = nn.KLDivLoss(reduction='none')
    batch_size = y.size(0)

    adv_probs = F.softmax(adv_output, dim=1)

    tmp1 = torch.argsort(adv_output, dim=1)[:, -2:]

    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    criterion = CrossEntropyLossMaybeSmooth(smooth_eps=config_dict['smooth_eps'])
    loss_adv = criterion(adv_output, y, smooth=config_dict['smooth']) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = torch.clamp(F.softmax(output, dim=1), min=1e-8)

    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(torch.sum(kl(torch.log(adv_probs+1e-12), nat_probs),dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(beta) * loss_robust

    return loss