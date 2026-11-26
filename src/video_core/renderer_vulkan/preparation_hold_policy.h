// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <mutex>

namespace Vulkan {

// FEAT-SGT-903: finishing the startup work is not the same as compiling every possible pipeline.
enum class PreparationHoldDecision { Wait, Drained, Deadline, Stalled };

constexpr PreparationHoldDecision DecidePreparationHold(bool wait_all, bool graphics_done,
                                                         bool translation_pending,
                                                         bool compute_pending, bool deadline,
                                                         bool stalled = false) {
    if (graphics_done && (!wait_all || (!translation_pending && !compute_pending))) {
        return PreparationHoldDecision::Drained;
    }
    if (wait_all && stalled) return PreparationHoldDecision::Stalled;
    if (!wait_all && deadline) return PreparationHoldDecision::Deadline;
    return PreparationHoldDecision::Wait;
}

// FEAT-SGT-904: measure inactivity, never the total preparation duration.
// A change in any queue counter or lifetime flag renews the observation window.
class PreparationStallWatch {
public:
    using Clock = std::chrono::steady_clock;
    using Progress = std::array<std::uint64_t, 8>;

    bool Observe(Progress progress, Clock::time_point now, Clock::duration limit) {
        std::scoped_lock lock{mutex};
        if (!initialized || progress != previous) {
            initialized = true;
            previous = progress;
            changed = now;
            return false;
        }
        return now - changed >= limit;
    }

private:
    std::mutex mutex;
    bool initialized{};
    Progress previous{};
    Clock::time_point changed{};
};

} // namespace Vulkan
