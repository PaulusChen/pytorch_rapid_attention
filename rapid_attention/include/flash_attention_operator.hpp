#include <type_traits>
#include <map>
#include <optional>
#include <torch/torch.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <tuple>
#include <type_traits>

namespace rapid_flash_attention {

::std::tuple<at::Tensor, float>
flash_attention_forward(const py::object &py_cfg, const torch::Tensor &TQ,
                        const torch::Tensor &TK, const torch::Tensor &TV,
                        ::std::optional<at::Tensor> &out_, bool benchmark);

} // namespace rapid_flash_attention