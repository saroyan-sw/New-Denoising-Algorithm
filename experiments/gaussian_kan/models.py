"""Distribution-free trainable generators. The target law is not used here."""

from efficient_kan import KAN


def make_kan(widths: list[int], grid_intervals: int, spline_order: int) -> KAN:
    if widths[0] != 1 or widths[-1] != 1:
        raise ValueError("generator must map one scalar to one scalar")
    return KAN(widths, grid_size=grid_intervals, spline_order=spline_order, grid_range=(-1.0, 1.0))


def parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
