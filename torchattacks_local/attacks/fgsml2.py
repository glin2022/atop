import torch
import torch.nn as nn

from ..attack import Attack


class FGSML2(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=8/255, eps_for_division=1e-10):
        super().__init__("FGSML2", model)
        self.eps = eps
        self.eps_for_division = eps_for_division
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        batch_size = len(images)
        adv_images = images.clone().detach()

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = self.get_logits(images)

        # Calculate loss
        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        grad_norms = torch.norm(grad.view(batch_size, -1), p=2, dim=1) + self.eps_for_division  # nopep8
        grad = grad / grad_norms.view(batch_size, 1, 1, 1)
        adv_images = adv_images.detach() + self.eps * grad

        delta = adv_images - images
        delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
        factor = self.eps / delta_norms
        factor = torch.min(factor, torch.ones_like(delta_norms))
        delta = delta * factor.view(-1, 1, 1, 1)
        
        
        adv_images = images + delta
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
