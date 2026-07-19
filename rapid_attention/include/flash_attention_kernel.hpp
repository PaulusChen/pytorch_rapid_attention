#pragma once

#include <map>
#include "flash_attention_config.h"

namespace rapid_flash_attention {
    extern ::std::map<FlashForwardKernelConfig, forward_kernel_fn> forward_kernels;
}