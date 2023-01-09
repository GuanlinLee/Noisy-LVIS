import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from functools import partial
from .accuracy import accuracy
from ..builder import LOSSES

def get_image_count_frequency(version="v0_5"):
    if version == "v0_5":
        from mmdet.utils.lvis_v0_5_categories import get_image_count_frequency
        return get_image_count_frequency()
    elif version == "v1":
        from mmdet.utils.lvis_v1_0_categories import get_image_count_frequency
        return get_image_count_frequency()
    elif version == "openimage":
        from mmdet.utils.openimage_categories import get_instance_count
        return get_instance_count()
    else:
        raise KeyError(f"version {version} is not supported")

@LOSSES.register_module()
class EQLv2(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=1203,  # 1203 for lvis v1.0, 1230 for lvis v0.5
                 gamma=12,
                 mu=0.8,
                 alpha=4.0,
                 vis_grad=False,
                 test_with_obj=True):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True
        self.custom_cls_channels = True
        self.custom_activation = True
        self.custom_accuracy = True
        # cfg for eqlv2
        self.vis_grad = vis_grad
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha
        self.freq_info = torch.FloatTensor(get_image_count_frequency('v1'))
        # initial variables
        self.register_buffer('pos_grad', torch.zeros(self.num_classes))
        self.register_buffer('neg_grad', torch.zeros(self.num_classes))
        # At the beginning of training, we set a high value (eg. 100)
        # for the initial gradient ratio so that the weight for pos gradients and neg gradients are 1.
        self.register_buffer('pos_neg', torch.ones(self.num_classes) * 100)

        self.test_with_obj = test_with_obj

        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))
        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)

    def get_cls_channels(self, num_classes):
        """Get custom classification channels.

        Args:
            num_classes (int): The number of classes.

        Returns:
            int: The custom classification channels.
        """
        assert num_classes == self.num_classes
        return num_classes + 1

    def get_accuracy(self, cls_score, labels):
        pos_inds = labels < self.num_classes
        acc_classes = accuracy(cls_score[pos_inds], labels[pos_inds])
        acc = dict()
        acc['acc_classes'] = acc_classes
        return acc

    def get_activation(self, cls_score):
        cls_score = torch.sigmoid(cls_score)
        n_i, n_c = cls_score.size()
        bg_score = cls_score[:, -1].view(n_i, 1)
        if self.test_with_obj:
            cls_score[:, :-1] *= (1 - bg_score)
        return cls_score

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        self.n_i, self.n_c = cls_score.size()

        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target

        target = expand_label(cls_score, label)

        pos_w, neg_w = self.get_weight(cls_score)

        weight = pos_w * target + neg_w * (1 - target)

        cls_loss = F.binary_cross_entropy_with_logits(cls_score, target,
                                                      reduction='none')
        cls_loss = torch.sum(cls_loss * weight) / self.n_i

        self.collect_grad(cls_score.detach(), target.detach(), weight.detach())

        return self.loss_weight * cls_loss

    def collect_grad(self, cls_score, target, weight):
        prob = torch.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)

        # do not collect grad for objectiveness branch [:-1]
        pos_grad = torch.sum(grad * target * weight, dim=0)[:-1]
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)[:-1]

        dist.all_reduce(pos_grad)
        dist.all_reduce(neg_grad)

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)

    def get_weight(self, cls_score):
        neg_w = torch.cat([self.map_func(self.pos_neg), cls_score.new_ones(1)])
        pos_w = 1 + self.alpha * (1 - neg_w)
        neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
        pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w