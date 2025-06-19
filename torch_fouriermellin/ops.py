from functools import lru_cache
import torch


class PhaseCorrelation(torch.nn.Module):
    def __init__(self, shift=True):
        super(PhaseCorrelation, self).__init__()
        self.shift = shift

    def forward(self, im, template):
        imFft = torch.fft.rfft2(im)
        templayteFft = torch.fft.rfft2(template)
        out = torch.fft.irfft2(
            (imFft * templayteFft.conj()) / (imFft * templayteFft).abs()
        )
        if not self.shift:
            return out
        return torch.fft.ifftshift(out)


class HighPassFilter(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @lru_cache(maxsize=1)
    def get_kernel(self, shape):
        ker = torch.outer(
            torch.blackman_window(shape[0]), torch.blackman_window(shape[1])
        )
        return torch.abs(ker - torch.max(ker)).unsqueeze(0).unsqueeze(0)

    def forward(self, img):
        return img.abs() * self.get_kernel(img.shape[-2:])
