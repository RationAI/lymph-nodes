from typing import Literal, TypeAlias, TypedDict

from torch import Tensor, nn

from lymph_nodes.typing import Outputs, Targets


LossKind: TypeAlias = Literal["labels", "masks"]


class CriterionLoss(TypedDict):
    weight: float
    name: str
    loss: nn.Module
    kind: LossKind


class SetCriterion(nn.Module):
    def __init__(self, losses: list[CriterionLoss]) -> None:
        super().__init__()
        self.losses = losses

    def loss(
        self,
        outputs: Outputs,
        targets: Targets,
    ) -> dict[str, Tensor]:
        # Compute all the requested losses
        losses = {
            loss["name"]: loss["loss"](
                getattr(outputs, loss["kind"]), getattr(targets, loss["kind"])
            )
            for loss in self.losses
        }

        losses["loss"] = sum(
            loss["weight"] * losses[loss["name"]] for loss in self.losses
        )

        return losses

    def forward(self, outputs: Outputs, targets: Targets) -> dict[str, Tensor]:
        # Compute all the requested losses
        return self.loss(outputs, targets)
