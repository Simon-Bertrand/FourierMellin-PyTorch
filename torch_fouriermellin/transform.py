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
        # aspect_x, aspect_y = 1, 1
        # if with_aspect:
        #     aspect_x, aspect_y = W / H, H / W

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


# class RigidTransform(torch.nn.Module):
#     def __init__(self, scale_alpha, trans_x, trans_y, rot_beta):
#         super().__init__()
#         self.scale_alpha = scale_alpha
#         self.trans_x = trans_x
#         self.trans_y = trans_y
#         self.rot_beta = rot_beta

#     @staticmethod
#     def rotmat_2d(rot: torch.Tensor) -> torch.Tensor:
#         cos_theta = torch.cos(rot / 180 * torch.pi)
#         sin_theta = torch.sin(rot / 180 * torch.pi)
#         R = torch.stack(
#             [
#                 torch.stack([cos_theta, sin_theta], dim=-1),
#                 torch.stack([-sin_theta, cos_theta], dim=-1),
#             ],
#             dim=-2,
#         )
#         return R

#     def forward(self, image):
#         # Apply a rigid (geometric) transform: scale, translation, rotation via bilinear interpolation
#         N, _, H, W = image.shape
#         transfMat = torch.cat(
#             [
#                 torch.cat(
#                     [
#                         self.scale_alpha.unsqueeze(-1).unsqueeze(-1)
#                         * self.rotmat_2d(self.rot_beta),
#                         torch.stack(
#                             [self.trans_x / (W // 2), self.trans_y / (H // 2)], dim=-1
#                         ).unsqueeze(-1),
#                     ],
#                     dim=-1,
#                 ),
#                 torch.cat(
#                     [
#                         torch.zeros(self.scale_alpha.size(0), 1, 2),
#                         torch.ones(self.scale_alpha.size(0), 1, 1),
#                     ],
#                     dim=-1,
#                 ),
#             ],
#             dim=-2,
#         )
#         coords = torch.stack(
#             torch.meshgrid(
#                 torch.linspace(-1, 1, W), torch.linspace(-1, 1, H), indexing="xy"
#             )
#         ).unsqueeze(0)
#         coordsHomo = torch.cat(
#             [coords.flatten(-2), torch.ones((coords.size(0), 1, H * W))], dim=-2
#         )
#         coordsTransf = (
#             torch.einsum("bcd,bdl->bcl", transfMat, coordsHomo)[..., :2, :]
#             .reshape(N, 2, H, W)
#             .moveaxis(1, -1)
#         )
#         return torch.nn.functional.grid_sample(
#             image,
#             coordsTransf,
#             mode="bilinear",
#             align_corners=True,
#         )
