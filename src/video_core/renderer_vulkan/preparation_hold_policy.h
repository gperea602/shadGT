// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

namespace Vulkan {

// FEAT-SGT-903: finishing the startup work is not the same as compiling every possible pipeline.
enum class PreparationHoldDecision { Wait, Drained, Deadline };

constexpr PreparationHoldDecision DecidePreparationHold(bool wait_all, bool graphics_done,
                                                         bool translation_pending,
                                                         bool compute_pending, bool deadline) {
    if (graphics_done && (!wait_all || (!translation_pending && !compute_pending))) {
        return PreparationHoldDecision::Drained;
    }
    if (!wait_all && deadline) return PreparationHoldDecision::Deadline;
    return PreparationHoldDecision::Wait;
}

} // namespace Vulkan
