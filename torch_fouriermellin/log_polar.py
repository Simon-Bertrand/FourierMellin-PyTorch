import math
import torch


class LogPolarRepresentation(torch.nn.Module):
    def __init__(self, H, W):
        super().__init__()
        self.H = H
        self.W = W

    def get_repr_size(self) -> tuple[int, int]:
        return 180, self.get_radius()

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

    def xy_grid(self):
        return torch.meshgrid(
            *[torch.linspace(1, -1, self.H), torch.linspace(1, -1, self.W)],
            indexing="xy",
        )

    def pol2cartgrid(self):
        radius = self.get_radius()
        indices = torch.complex(*self.xy_grid())
        return (radius * indices.abs() / math.sqrt(2)).log() / math.log(
            radius
        ) * 2 - 1, indices.angle() / torch.pi

    def thetarho_grid(self):
        radius = (
            (torch.norm(torch.tensor([self.H, self.W], dtype=torch.float32)) / 2)
            .long()
            .item()
        )
        reprH, reprW = self.get_repr_size()
        theta, r = torch.meshgrid(
            *[torch.arange(2 * reprH), torch.arange(reprW)], indexing="ij"
        )
        return theta, radius ** (r / radius)

    def cart2polgrid(self):
        theta, rho = self.thetarho_grid()
        indices = torch.polar(rho, theta * torch.pi / 180)
        return indices.real + self.W // 2, indices.imag + self.H // 2

    def cart2pol(self, img):
        xInds, yInds = self.cart2polgrid()
        grid = torch.stack(
            [
                (xInds / (img.shape[-2] - 1)) * 2 - 1,
                (yInds / (img.shape[-1] - 1)) * 2 - 1,
            ],
            dim=-1,
        )
        return self.remap(img, grid)

    def pol2cart(self, img):
        grid = torch.stack(self.pol2cartgrid(), dim=-1)
        return self.remap(img, grid)
