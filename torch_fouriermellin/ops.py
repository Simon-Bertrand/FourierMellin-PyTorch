from functools import lru_cache
import torch


class PhaseCorrelation(torch.nn.Module):
    def __init__(self, shift=True):
        super(PhaseCorrelation, self).__init__()
        self.shift = shift

    def forward(self, im, template):

        imFft = torch.fft.rfft2(im, s=im.shape[-2:])
        templateFft = torch.fft.rfft2(template, s=im.shape[-2:])
        out = torch.fft.irfft2(
            (imFft * templateFft.conj()) / (imFft * templateFft).abs(), s=im.shape[-2:]
        )
        if not self.shift:
            return out
        return torch.fft.ifftshift(out)


class HighPassFilter(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @lru_cache(maxsize=1)
    def get_kernel(self, shape, device=None):
        ker = torch.outer(
            torch.blackman_window(shape[0], device=device),
            torch.blackman_window(shape[1], device=device),
        )
        return torch.abs(ker - torch.max(ker)).unsqueeze(0).unsqueeze(0)

    def forward(self, img):
        return img.abs() * self.get_kernel(img.shape[-2:], device=img.device)
