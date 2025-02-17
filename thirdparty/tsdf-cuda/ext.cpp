#include <torch/extension.h>
#include "integrate_kernel.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("integrateTSDFVolume", &integrateTSDFVolume);
}
