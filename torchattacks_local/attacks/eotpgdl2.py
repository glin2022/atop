import torch
import torch.nn as nn

from ..attack import Attack


class EOTPGDL2(Attack):
    r"""
    Comment on "Adv-BNN: Improved Adversarial Defense through Robust Bayesian Neural Network"
    [https://arxiv.org/abs/1907.00895]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        eot_iter (int) : number of models to estimate the mean gradient. (Default: 2)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.EOTPGD(model, eps=8/255, alpha=2/255, steps=10, eot_iter=2)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8/255, alpha=2/255, steps=10,
                 eot_iter=2, random_start=True, eps_for_division=1e-10):
        super().__init__("EOTPGDL2", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.eot_iter = eot_iter
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']
        self.eps_for_division = eps_for_division

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()
        batch_size = len(images)

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)  # nopep8
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            grad = torch.zeros_like(adv_images)
            adv_images.requires_grad = True

            for j in range(self.eot_iter):
                outputs = self.get_logits(adv_images)

                # Calculate loss
                if self.targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)

                # Update adversarial images
                grad += torch.autograd.grad(cost, adv_images,
                                            retain_graph=False,
                                            create_graph=False)[0]

            # (grad/self.eot_iter).sign() == grad.sign()
            grad_norms = torch.norm(grad.view(batch_size, -1), p=2, dim=1) + self.eps_for_division  # nopep8
            grad = grad / grad_norms.view(batch_size, 1, 1, 1)
            adv_images = adv_images.detach() + self.alpha * grad

            delta = adv_images - images
            delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
            factor = self.eps / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(-1, 1, 1, 1)

            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
