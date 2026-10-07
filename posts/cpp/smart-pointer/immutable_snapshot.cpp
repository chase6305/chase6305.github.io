// C++20: immutable latest-state publication; NOT a lossless or hard-real-time queue.
#include <atomic>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <thread>
#include <version>

#if !defined(__cpp_lib_atomic_shared_ptr) || __cpp_lib_atomic_shared_ptr < 201711L
#error "This example needs a C++20 library with atomic shared_ptr (e.g. libstdc++ 12+)."
#endif

struct Snapshot {
    std::uint64_t sequence;
    double position;
    double negative_position;
};

int main() {
    std::atomic<std::shared_ptr<const Snapshot>> latest{nullptr};
    // A local owning copy keeps the OLD object alive after publication changes.
    latest.store(std::make_shared<const Snapshot>(Snapshot{1, .125, -.125}),
                 std::memory_order_release);
    auto held = latest.load(std::memory_order_acquire);
    std::weak_ptr<const Snapshot> old = held;
    latest.store(std::make_shared<const Snapshot>(Snapshot{2, .25, -.25}),
                 std::memory_order_release);
    assert(!old.expired() && held->sequence == 1);
    held.reset();
    assert(old.expired());
    latest.store(nullptr, std::memory_order_release);

    constexpr std::uint64_t publications = 100000;
    std::atomic<bool> closed{false};
    std::uint64_t last_seen = 0;
    std::uint64_t distinct_observed = 0;
    const bool lock_free = latest.is_lock_free();
    std::thread reader([&] {
        for (;;) {
            // Keep this shared_ptr for the entire read. A temporary load().get()
            // would leave only a raw pointer once the temporary is destroyed.
            auto snapshot = latest.load(std::memory_order_acquire);
            if (snapshot) {
                assert(snapshot->sequence >= last_seen);
                assert(snapshot->position == .125 * snapshot->sequence);
                assert(snapshot->negative_position == -snapshot->position);
                if (snapshot->sequence != last_seen) ++distinct_observed;
                last_seen = snapshot->sequence;
            }
            if (closed.load(std::memory_order_acquire) && last_seen == publications) break;
            std::this_thread::yield();
        }
    });
    std::thread writer([&] {
        for (std::uint64_t sequence = 1; sequence <= publications; ++sequence) {
            const double position = .125 * sequence;
            // The allocated object itself is const, with no mutable alias.
            latest.store(std::make_shared<const Snapshot>(
                Snapshot{sequence, position, -position}), std::memory_order_release);
        }
        closed.store(true, std::memory_order_release);
    });
    writer.join();
    reader.join();
    assert(last_seen == publications);
    assert(distinct_observed >= 1 && distinct_observed <= publications);
    std::cout << "PASS: coherent immutable snapshots and retained ownership; published="
              << publications << ", distinct observed=" << distinct_observed
              << ", atomic shared_ptr lock-free=" << std::boolalpha << lock_free << '\n';
}
