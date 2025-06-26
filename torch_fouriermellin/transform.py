import torch


class RigidTransform(torch.nn.Module):
    def __init__(self, scale_alpha, trans_x, trans_y, rot_beta, device=None):
        super().__init__()
        self.scale_alpha = scale_alpha.to(device)
        self.trans_x = trans_x.to(device)
        self.trans_y = trans_y.to(device)
        self.rot_beta = rot_beta.to(device)

    @staticmethod
    def rotmat_2d(rot: torch.Tensor) -> torch.Tensor:
        cos_theta = torch.cos(rot / 180 * torch.pi)
        sin_theta = torch.sin(rot / 180 * torch.pi)
        R = torch.stack(
            [
                torch.stack([cos_theta, sin_theta], dim=-1),
                torch.stack([-sin_theta, cos_theta], dim=-1),
            ],
            dim=-2,
        )
        return R

    def forward(self, image):
        # Apply a rigid (geometric) transform: scale, translation, rotation via bilinear interpolation
        N, _, H, W = image.shape
        R = self.rotmat_2d(self.rot_beta) * self.scale_alpha.view(-1, 1, 1)
        translation = torch.stack(
            [self.trans_x * 2 / W, self.trans_y * 2 / H], dim=-1
        ).unsqueeze(-1)

        return torch.nn.functional.grid_sample(
            image,
            torch.nn.functional.affine_grid(
                torch.cat([R, translation], dim=-1), image.size(), align_corners=True
            ),
            mode="bilinear",
            align_corners=True,
        )

