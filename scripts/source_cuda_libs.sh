#!/usr/bin/env bash
# After: source .venv/bin/activate
# Run:   source /path/to/sparx/scripts/source_cuda_libs.sh
#
# PyTorch/torchaudio wheels link to libcudart.so.12; NVIDIA ships it under
# site-packages/nvidia/*/lib. The loader often does not see those dirs unless
# LD_LIBRARY_PATH includes them (common when no full system CUDA toolkit).

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "source_cuda_libs.sh: activate your venv first (VIRTUAL_ENV is empty)." >&2
  return 1 2>/dev/null || exit 1
fi

_extra="$(python3 -c "
import glob, os, site
roots = site.getsitepackages()
if not roots:
    raise SystemExit('no site-packages')
p = roots[0]
paths = []
for d in glob.glob(os.path.join(p, 'nvidia', '*', 'lib')):
    if os.path.isdir(d):
        paths.append(d)
print(':'.join(paths))
")"

if [[ -n "$_extra" ]]; then
  export LD_LIBRARY_PATH="${_extra}:${LD_LIBRARY_PATH:-}"
  echo "source_cuda_libs.sh: prepended ${_extra} to LD_LIBRARY_PATH"
else
  echo "source_cuda_libs.sh: no venv site-packages/nvidia/*/lib found; try: pip install nvidia-cuda-runtime-cu12" >&2
fi
unset _extra
