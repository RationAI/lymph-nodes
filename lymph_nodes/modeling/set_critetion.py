from torch import Tensor, nn


class SetCriterion(nn.Module):
    def __init__(
        self,
        weight_dict: dict[str, float],
        losses: dict[str, nn.Module],
    ) -> None:
        """Create the criterion.

        Args:
            weight_dict: Dict containing as key the names of the losses and as values their relative weight.
        """
        super().__init__()
        self.weight_dict = weight_dict

        # losses
        self.losses = losses

    def loss(
        self,
        outputs: Tensor,
        targets: Tensor,
    ) -> dict[str, Tensor]:
        """Compute the losses.

        Args:
            outputs: predictions of the model
            targets: ground truth masks
        """

        # Compute all the requested losses
        losses = {k: v(outputs, targets) for k, v in self.losses.items()}

        losses["loss"] = sum(w * losses[k] for k, w in self.weight_dict.items())

        return losses

    def forward(self, outputs: Tensor, targets: Tensor) -> dict[str, Tensor]:
        """This performs the loss computation.

        Args:
            outputs: predictions of the model
            targets: ground truth masks
        """

        # Compute all the requested losses
        return self.loss(outputs, targets)
