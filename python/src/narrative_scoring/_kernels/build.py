"""Build the fused scoring kernel in place.

    uv run python -m narrative_scoring._kernels.build

Reproducible on any machine with a C compiler: OpenMP is used when the
toolchain supports it and silently dropped when it does not (macOS/clang
without libomp), in which case the kernel still runs, single-threaded.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _openmp_available(cc: str) -> bool:
    src = "#include <omp.h>\nint main(void){return omp_get_max_threads();}\n"
    with tempfile.TemporaryDirectory() as tmp:
        c = Path(tmp) / "probe.c"
        c.write_text(src)
        return subprocess.run(
            [cc, "-fopenmp", str(c), "-o", str(Path(tmp) / "probe")],
            capture_output=True,
        ).returncode == 0


def main() -> int:
    try:
        from Cython.Build import cythonize
        from setuptools import Extension, setup
        import numpy as np
    except ImportError as exc:
        print(f"cannot build: {exc}", file=sys.stderr)
        return 1

    cc = os.environ.get("CC", "cc")
    omp = _openmp_available(cc)
    # NO -ffast-math: it permits reassociating
    #     ceil(n_surv * (100 - q) / 100)
    # into  n_surv * (5.0/100.0), and 0.05 is not representable, so
    # 140 * 0.05 = 7.000000000000000388 -> ceil 8 instead of 7. That shifted the
    # percentile cut by one position on every row whose survivor count divides
    # exactly (n_surv % 20 == 0 at q=95), silently changing the panel.
    # Reproducibility beats the few percent it buys on comparison-bound loops.
    flags = ["-O3", "-fno-fast-math", "-march=native"]
    link: list[str] = []
    if omp:
        flags.append("-fopenmp")
        link.append("-fopenmp")
    else:
        print("OpenMP not available; building single-threaded", file=sys.stderr)

    ext = Extension(
        "narrative_scoring._kernels.fused_gate",
        sources=[str(HERE / "fused_gate.pyx")],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=flags + ["-std=c++17"],
        extra_link_args=link,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
    sys.argv = [sys.argv[0], "build_ext", "--inplace"]
    setup(
        name="narrative_scoring_kernels",
        ext_modules=cythonize([ext], language_level=3, quiet=True),
        script_args=["build_ext", "--inplace"],
        options={"build_ext": {"build_lib": str(HERE.parent.parent)}},
    )
    print(f"built with openmp={omp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
