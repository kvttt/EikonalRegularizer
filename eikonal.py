import torch
import torch.nn as nn
import torch.nn.functional as F


class EikonalRegularizer(nn.Module):
    """
    Eikonal regularization for signed distance fields (SDFs).
    """

    def __init__(self, spacing: tuple[float, ...], reduction: str = 'mean') -> None:
        """
        Constructs the Eikonal regularizer.

        Parameters
        ----------
        spacing : tuple[float, ...]
            The spacing between grid points in each dimension, expected to be a tuple of 1, 2, or 3 elements.
        reduction : str, optional
            Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'. Default is 'mean'.

        Returns
        -------
        None
        """

        super(EikonalRegularizer, self).__init__()

        if len(spacing) not in [1, 2, 3]:
            raise ValueError(f"Expect spacing to be a tuple of 1, 2, or 3 elements, got {len(spacing)} elements")

        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Expect reduction to be one of 'none', 'mean', or 'sum', got {reduction}")

        self.spacing = spacing
        self.reduction = reduction

    def forward(self, sdf: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the Eikonal regularization loss for a signed distance field (SDF).

        Parameters
        ----------
        sdf : torch.Tensor
            The signed distance field tensor, expected to be 3D, 4D, or 5D.
        mask : torch.Tensor, optional
            A mask tensor that indicates valid regions in the SDF. Default is None.

        Returns
        -------
        loss : torch.Tensor
            The computed Eikonal loss.
        grad_norm : torch.Tensor
            The gradient norm of the SDF.
        """

        if sdf.ndim not in [3, 4, 5]:
            raise ValueError(f"Expect SDF to be 3D, 4D, or 5D tensor, got {sdf.ndim}D tensor")

        if not sdf.ndim == len(self.spacing) + 2:
            raise ValueError(f"Expect sdf to have {len(self.spacing) + 2} dimensions, got {sdf.ndim} dimensions")

        if mask is None:
            mask = torch.ones_like(sdf, dtype=torch.bool)

        if not mask.shape == sdf.shape:
            raise ValueError(
                f"Expect mask to have the same shape as sdf, got mask shape {mask.shape} and sdf shape {sdf.shape}"
            )

        masking = mask.bool().float()

        sdf_padded = F.pad(sdf, pad=(1, 1) * (sdf.ndim - 2), mode='replicate')

        if sdf.ndim == 3:
            dx = self.spacing[0]
            dfdx = (sdf_padded[:, :, 2:] - sdf_padded[:, :, :-2]) / (2 * dx)
            grad_norm = torch.sqrt(dfdx ** 2)
        elif sdf.ndim == 4:
            dx, dy = self.spacing[0], self.spacing[1]
            dfdx = (sdf_padded[:, :, 2:, 1:-1] - sdf_padded[:, :, :-2, 1:-1]) / (2 * dx)
            dfdy = (sdf_padded[:, :, 1:-1, 2:] - sdf_padded[:, :, 1:-1, :-2]) / (2 * dy)
            grad_norm = torch.sqrt(dfdx ** 2 + dfdy ** 2)
        else:
            dx, dy, dz = self.spacing[0], self.spacing[1], self.spacing[2]
            dfdx = (sdf_padded[:, :, 2:, 1:-1, 1:-1] - sdf_padded[:, :, :-2, 1:-1, 1:-1]) / (2 * dx)
            dfdy = (sdf_padded[:, :, 1:-1, 2:, 1:-1] - sdf_padded[:, :, 1:-1, :-2, 1:-1]) / (2 * dy)
            dfdz = (sdf_padded[:, :, 1:-1, 1:-1, 2:] - sdf_padded[:, :, 1:-1, 1:-1, :-2]) / (2 * dz)
            grad_norm = torch.sqrt(dfdx ** 2 + dfdy ** 2 + dfdz ** 2)

        if self.reduction == 'none':
            loss = (grad_norm - 1.0 * masking) ** 2
        elif self.reduction == 'mean':
            loss = ((grad_norm - 1.0 * masking) ** 2).mean()
        else:
            loss = ((grad_norm - 1.0 * masking) ** 2).sum()

        return loss, grad_norm
