"""Optional single-kernel BGR uint8 -> normalized NCHW preprocessing."""

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _bgr_to_nchw(source, destination, PIXELS: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < PIXELS * 3
        channel = offsets // PIXELS
        pixel = offsets % PIXELS
        values = tl.load(source + pixel * 3 + (2 - channel), mask=valid, other=0)
        tl.store(destination + offsets, values.to(tl.float32) * (1.0 / 255.0), mask=valid)


def fused_preprocess(source, destination):
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    pixels = source.shape[0] * source.shape[1]
    _bgr_to_nchw[(triton.cdiv(pixels * 3, 256),)](source, destination, pixels, 256)
