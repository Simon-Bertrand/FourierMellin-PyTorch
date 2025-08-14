import torch

from .log_polar import (
    LogPolarRepresentation,
)
from .ops import (
    HighPassFilter,
    PhaseCorrelation,
)
from .transform import RigidTransform


class MellinFourierRegistration(torch.nn.Module):
    def __init__(self, H, W):
        super().__init__()
        self.logPolar = LogPolarRepresentation(H, W)
        self.highPassFilter = HighPassFilter()

    def transform_rot_parameter(self, rotIdx):
        return -rotIdx / self.logPolar.angular_space_length * 360 + 90

    def transform_scale_parameter(self, scaleIdx, width):
        radius = self.logPolar.get_radius()
        return radius ** (-((scaleIdx - width // 2)) / radius)

    def transform_translation_parameters(self, iMax, jMax, pcH, pcW):
        return torch.stack(
            (
                iMax.where(iMax < pcH // 2, iMax - pcH),
                jMax.where(jMax < pcW // 2, jMax - pcW),
            ),
            dim=-1,
        )

    def get_rot_scale(
        self,
        pc_rot_scale,
    ):
        topPc = pc_rot_scale.abs().flatten(-2).topk(1, dim=(-1))
        predsRotsScale = torch.stack(
            torch.unravel_index(topPc.indices, pc_rot_scale.shape[-2:]), -2
        )
        estRot = self.transform_rot_parameter(predsRotsScale[:, 0, 0])
        estScale = self.transform_scale_parameter(
            predsRotsScale[:, 1], pc_rot_scale.size(-1)
        )[..., 0]
        return estRot, estScale

    def get_translation_dirac(self, H, W, tx, ty):
        assert ty.size(0) == ty.size(
            0
        ), "Batch size mismatch in translation parameters."
        dirac = torch.zeros(ty.size(0), H, W, dtype=torch.bool, device=ty.device)
        dirac[torch.arange(ty.size(0)), ty, tx] = True
        return dirac

    def get_translations(self, pc_translat):
        iMax, jMax = torch.unravel_index(
            pc_translat.flatten(-2).sum(-2).argmax(-1), pc_translat.shape[-2:]
        )
        estTrans = self.transform_translation_parameters(
            iMax, jMax, *pc_translat.shape[-2:]
        )

        return estTrans

    def forward(self, image, template):
        imageFft = torch.fft.fftshift(torch.fft.fft2(image))
        templateFft = torch.fft.fftshift(torch.fft.fft2(template))
        imageFftAbs = self.highPassFilter(imageFft)
        templateFftAbs = self.highPassFilter(templateFft)
        imageLogPolar = self.logPolar.cart2pol(imageFftAbs)
        templateLogPolar = self.logPolar.cart2pol(templateFftAbs)
        pcRotScale = PhaseCorrelation(shift=True)(
            imageLogPolar[..., : self.logPolar.angular_space_length // 2, :],
            templateLogPolar[..., : self.logPolar.angular_space_length // 2, :],
        ).sum(-3)
        estRot, estScale = self.get_rot_scale(pcRotScale)
        templateUnrotUnscaled = RigidTransform(
            1 / estScale,
            torch.zeros_like(estScale, device=image.device),
            torch.zeros_like(estScale, device=image.device),
            -estRot,
            device=image.device,
        )(template)
        if torch.isnan(templateUnrotUnscaled).any():
            raise ValueError("NaN detected in templateUnrotUnscaled")
        if torch.isnan(image).any():
            raise ValueError("NaN detected in image")
        pcTranslat = PhaseCorrelation(shift=False)(image, templateUnrotUnscaled)
        estTrans = self.get_translations(pcTranslat)
        return {
            "estRot": estRot,
            "estScale": estScale,
            "estTrans": estTrans,
            "pcRotScale": pcRotScale,
            "pcTranslat": pcTranslat,
        }

    def register_image(self, image, template):
        if image.shape[-2] != image.shape[-1]:
            raise RuntimeError(
                "Non-square images are not supported in MellinFourierRegistration. This feature is not implemented currently."
            )
        params = self(image, template)
        estTransTransf = torch.einsum(
            "bcd, bd -> bc",
            RigidTransform.rotmat_2d(params["estRot"]),
            params["estTrans"].float(),
        ) / params["estScale"].unsqueeze(-1)
        imageTransfInv = RigidTransform(
            1 / params["estScale"],
            -estTransTransf[:, 1],
            -estTransTransf[:, 0],
            -params["estRot"],
            device=image.device,
        )(template)
        return dict(registered=imageTransfInv, params=params)

    def get_parameters_domain(self):
        pcRotScaleSize = (
            self.logPolar.angular_space_length // 2,
            self.logPolar.get_radius(),
        )
        randomRotScale = torch.unravel_index(
            torch.arange(pcRotScaleSize[0] * pcRotScaleSize[1]), pcRotScaleSize
        )
        pcTranslatH, pcTranslatW = self.logPolar.H, self.logPolar.W
        randomTranslat = torch.unravel_index(
            torch.arange(pcTranslatH * pcTranslatW), (pcTranslatH, pcTranslatW)
        )
        randomRotScale = torch.stack(
            [
                self.transform_rot_parameter(randomRotScale[0]),
                self.transform_scale_parameter(randomRotScale[1], pcRotScaleSize[1]),
            ],
            -1,
        )
        randomTrans = self.transform_translation_parameters(
            randomTranslat[0], randomTranslat[1], pcTranslatH, pcTranslatW
        )
        return dict(
            tyRange=randomTrans[:, 0].aminmax(),
            txRange=randomTrans[:, 1].aminmax(),
            rotRange=randomRotScale[:, 0].aminmax(),
            scaleRange=randomRotScale[:, 1].aminmax(),
        )

    def get_random_parameters(self, N):
        rotMaxIdx, scaleMaxIdx = self.logPolar.get_repr_size()
        randomRotIdx = torch.randint(0, rotMaxIdx, (N,))
        randomScaleIdx = torch.randint(0, scaleMaxIdx, (N,))
        randomTransYIdx = torch.randint(0, self.logPolar.H, (N,))
        randomTransXIdx = torch.randint(0, self.logPolar.W, (N,))
        rotScaleGrid = torch.meshgrid(
            [torch.arange(0, rotMaxIdx), torch.arange(0, scaleMaxIdx)], indexing="ij"
        )
        transGrid = torch.meshgrid(
            [torch.arange(0, self.logPolar.H), torch.arange(0, self.logPolar.W)],
            indexing="ij",
        )
        pcRotScaleTruthMask = (
            rotScaleGrid[0].unsqueeze(0) == randomRotIdx.unsqueeze(-1).unsqueeze(-1)
        ) & (rotScaleGrid[1].unsqueeze(0) == randomScaleIdx.unsqueeze(-1).unsqueeze(-1))
        pcTransTruthMask = (
            transGrid[0].unsqueeze(0) == randomTransYIdx.unsqueeze(-1).unsqueeze(-1)
        ) & (transGrid[1].unsqueeze(0) == randomTransXIdx.unsqueeze(-1).unsqueeze(-1))
        randomRotScale = torch.stack(
            [
                self.transform_rot_parameter(randomRotIdx),
                self.transform_scale_parameter(randomScaleIdx, scaleMaxIdx),
            ],
            -1,
        )
        randomTrans = self.transform_translation_parameters(
            randomTransYIdx, randomTransXIdx, self.logPolar.H, self.logPolar.W
        )
        return dict(
            gtRot=randomRotScale[:, 0],
            gtScale=randomRotScale[:, 1],
            gtTrans=randomTrans,
            gtPcRotScaleMask=pcRotScaleTruthMask,
            gtPcTransMask=pcTransTruthMask,
        )


def scale_loss(estScale, gtScale):
    return ((estScale / gtScale - 1) ** 2).sum(-1)


def rot_loss(estRot, gtAngle):
    return ((((estRot) - gtAngle) / 180) ** 2).sum(-1)
