"""Device-resident ITK v4 window convergence for independent frame losses."""

import torch


class WindowConvergence:
    """Track ITK's normalized, first-order B-spline energy-profile gradient.

    Callers select finite, active losses. Stopped frames retain their
    history and accumulated absolute energy; no tensors are read on the host.

    Parameters
    ----------
    batch_size : int
        Number of independent frames.
    window_size : int
        Number of active loss samples in each convergence window.
    device : torch.device
        Device holding the per-frame histories and convergence values.
    """

    def __init__(self, batch_size: int, window_size: int, device: torch.device):
        """Initialize spline weights and device-resident per-frame histories."""
        # ITK's spline fitter uses float32 coordinates/weights, even for double energies.
        position = torch.arange(window_size, dtype=torch.float32, device="cpu")
        position /= window_size - 1
        position[-1] = 1 - torch.tensor(1e-4, dtype=torch.float32)
        basis = 1 - (position[:, None] - torch.arange(2, device="cpu")).abs()
        squared = basis.square()
        omega = torch.zeros(2, dtype=torch.float32, device="cpu")
        for row in squared:
            omega += row
        factors = squared * basis / squared.sum(dim=1, keepdim=True)
        # The fixed fit and endpoint derivative are linear in the window energies.
        coefficients = factors.double() / omega.double()
        self.weights = (coefficients[:, 0] - coefficients[:, 1]).to(
            device=device, non_blocking=True
        )
        self.history = torch.zeros(
            (batch_size, window_size), dtype=torch.float64, device=device
        )
        self.total = self.history.new_zeros(batch_size)
        self.count = torch.zeros(batch_size, dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, loss: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """Return convergence values, or double max until a full window exists.

        Parameters
        ----------
        loss : torch.Tensor
            Per-frame objective values, with shape (batch_size,).
        active : torch.Tensor
            Boolean mask selecting frames whose histories should advance.

        Returns
        -------
        torch.Tensor
            Float64 convergence values, with the largest finite double used
            until a frame has accumulated a complete window.
        """
        self.history = torch.where(
            active[:, None],
            torch.cat((self.history[:, 1:], loss.detach()[:, None]), dim=1),
            self.history,
        )
        self.total += torch.where(active, loss.detach().abs(), 0)
        self.count += active
        normalized = self.history / torch.where(self.total > 0, self.total, 1)[:, None]
        value = (normalized * self.weights).sum(dim=1)
        return torch.where(
            self.count >= self.history.shape[1], value, torch.finfo(torch.float64).max
        )
