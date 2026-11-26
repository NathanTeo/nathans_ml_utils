# ========================================
# Imports
import contextlib
import warnings
from typing import Any, Generator, Literal

import torch
from torch.jit._trace import TracerWarning

from nathans_ml_utils import dnnlib

# ========================================
# Symbolic assert

try:
    symbolic_assert = torch._assert  # 1.8.0a0 # pylint: disable=protected-access
except AttributeError:
    symbolic_assert = torch.Assert  # 1.7.0

# ========================================
# Functions


@contextlib.contextmanager
def suppress_tracer_warnings() -> Generator[None, Any, None]:
    """
    Context manager to temporarily suppress known warnings in torch.jit.trace().
    Note: Cannot use catch_warnings because of https://bugs.python.org/issue29672
    """
    flt: tuple[Literal["ignore"], None, type[TracerWarning], None, Literal[0]] = (
        "ignore",
        None,
        torch.jit.TracerWarning,
        None,
        0,
    )
    warnings.filters.insert(0, flt)
    yield
    warnings.filters.remove(flt)


def assert_shape(tensor: torch.Tensor, ref_shape: torch.Tensor):
    """
    Assert that the shape of a tensor matches the given list of integers.
    None indicates that the size of a dimension is allowed to vary.
    Performs symbolic assertion when used in torch.jit.trace().
    """
    if tensor.ndim != len(ref_shape):
        raise AssertionError(
            f"Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}"
        )
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            with (
                suppress_tracer_warnings()
            ):  # as_tensor results are registered as constants
                symbolic_assert(
                    torch.equal(torch.as_tensor(size), ref_size),
                    f"Wrong size for dimension {idx}",
                )
        elif isinstance(size, torch.Tensor):
            with (
                suppress_tracer_warnings()
            ):  # as_tensor results are registered as constants
                symbolic_assert(
                    torch.equal(size, torch.as_tensor(ref_size)),
                    f"Wrong size for dimension {idx}: expected {ref_size}",
                )
        elif size != ref_size:
            raise AssertionError(
                f"Wrong size for dimension {idx}: got {size}, expected {ref_size}"
            )


def print_module_summary(
    module: torch.nn.Module,
    inputs: tuple | list,
    max_nesting: int = 3,
    skip_redundant: bool = True,
) -> Any:
    """Print summary table of module heirachy"""
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries: list = []
    nesting: list[int] = [0]

    def pre_hook(_mod, _inputs):
        nesting[0] += 1

    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(dnnlib.EasyDict(mod=mod, outputs=outputs))

    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {
            id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs
        }

    # Filter out redundant entries.
    if skip_redundant:
        entries = [
            e
            for e in entries
            if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)
        ]

    # Construct table.
    rows = [
        [type(module).__name__, "Parameters", "Buffers", "Output shape", "Datatype"]
    ]
    rows += [["---"] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = "<top-level>" if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(t.shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split(".")[-1] for t in e.outputs]
        rows += [
            [
                name + (":0" if len(e.outputs) >= 2 else ""),
                str(param_size) if param_size else "-",
                str(buffer_size) if buffer_size else "-",
                (output_shapes + ["-"])[0],
                (output_dtypes + ["-"])[0],
            ]
        ]
        for idx in range(1, len(e.outputs)):
            rows += [
                [name + f":{idx}", "-", "-", output_shapes[idx], output_dtypes[idx]]
            ]
        param_total += param_size
        buffer_total += buffer_size
    rows += [["---"] * len(rows[0])]
    rows += [["Total", str(param_total), str(buffer_total), "-", "-"]]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print(
            "  ".join(
                cell + " " * (width - len(cell)) for cell, width in zip(row, widths)
            )
        )
    print()
    return outputs
