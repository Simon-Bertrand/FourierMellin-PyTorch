import torch
import math


class LogPolarRepresentation(torch.nn.Module):
    def __init__(self, H, W, angular_space_length=None):
        super().__init__()
        self.H = H
        self.W = W
        self.angular_space_length = (
            self.get_radius() if angular_space_length is None else angular_space_length
        )
        if self.W != self.H:
            raise NotImplementedError(
                "Image must be square for polar to cartesian conversion."
            )

    def get_repr_size(self) -> tuple[int, int]:
        return self.angular_space_length, self.get_radius()

    def get_radius(self) -> int:
        return int(
            (torch.norm(torch.tensor([self.H, self.W], dtype=torch.float32)) / 2).item()
        )

    def remap(self, img, grid):
        return torch.nn.functional.grid_sample(
            img,
            grid.clamp(-1, 1).unsqueeze(0).expand(img.size(0), *3 * [-1]),
            mode="bilinear",
            align_corners=True,
        )

    def xy_grid(self, device):
        return torch.meshgrid(
            *[
                torch.linspace(1, -1, self.W, device=device),
                torch.linspace(1, -1, self.H, device=device),
            ],
            indexing="xy",
        )

    def pol2cartgrid(self, device):
        radius = self.get_radius()
        indices = torch.complex(*self.xy_grid(device=device))
        return (radius * indices.abs() / math.sqrt(2)).log() / math.log(
            radius
        ) * 2 - 1, indices.angle() / torch.pi

    def thetarho_grid(self, device=None):
        radius = self.get_radius()
        reprH, reprW = self.get_repr_size()
        theta, r = torch.meshgrid(
            *[
                torch.linspace(
                    0,
                    360,
                    reprH,
                    device=device,
                    dtype=torch.float32,
                ),
                torch.arange(reprW, device=device, dtype=torch.float32),
            ],
            indexing="ij",
        )
        return theta, radius ** (r / radius)

    def cart2polgrid(self, device):
        theta, rho = self.thetarho_grid(device=device)
        indices = torch.polar(rho, theta * torch.pi / 180)
        return indices.real + self.W / 2, indices.imag + self.H / 2

    def cart2pol(self, img):
        xInds, yInds = self.cart2polgrid(img.device)
        grid = torch.stack(
            [
                (xInds / (img.shape[-1] - 1)) * 2 - 1,
                (yInds / (img.shape[-2] - 1)) * 2 - 1,
            ],
            dim=-1,
        )
        return self.remap(img, grid)

    def pol2cart(self, img):

        grid = torch.stack(self.pol2cartgrid(img.device), dim=-1)
        return self.remap(img, grid)
