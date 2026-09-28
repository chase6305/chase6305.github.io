// Linux / C++17. Local socketpair checks; no robot or external network access.
#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
constexpr std::uint32_t max_payload = 1024 * 1024;

struct Timeout : std::runtime_error {
    Timeout() : std::runtime_error("frame deadline expired") {}
};

void check_deadline(Clock::time_point deadline) {
    if (Clock::now() >= deadline) throw Timeout();
}

// False means EOF before ANY requested byte. Partial EOF is an error.
bool read_exact_until(int fd, char* data, std::size_t size,
                      Clock::time_point deadline) {
    std::size_t received = 0;
    while (received < size) {
        const auto now = Clock::now();
        if (now >= deadline) throw Timeout();
        // poll rounds to milliseconds: round up, then recheck the real clock.
        const auto remaining = std::chrono::ceil<std::chrono::milliseconds>(deadline - now);
        const int wait_ms = static_cast<int>(std::min<long long>(remaining.count(), INT_MAX));
        pollfd event{fd, POLLIN, 0};
        const int ready = ::poll(&event, 1, wait_ms);
        if (ready < 0 && errno == EINTR) continue;
        if (ready < 0) throw std::system_error(errno, std::generic_category(), "poll");
        if (ready == 0) continue;
        if (event.revents & POLLNVAL)
            throw std::system_error(EBADF, std::generic_category(), "poll descriptor");
        check_deadline(deadline);
        // Readiness is not a reservation. This call itself must not block.
        const auto n = ::recv(fd, data + received, size - received, MSG_DONTWAIT);
        if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (n < 0) throw std::system_error(errno, std::generic_category(), "recv");
        check_deadline(deadline);
        if (n == 0) {
            if (received == 0) return false;
            throw std::runtime_error("truncated frame");
        }
        received += static_cast<std::size_t>(n);
    }
    check_deadline(deadline);
    return true;
}

// nullopt = clean EOF at frame boundary; empty string = a valid zero-byte frame.
// On any exception, discard this connection: consumed partial bytes are not saved.
std::optional<std::string> read_frame_until(int fd, Clock::time_point deadline) {
    std::uint32_t network_length = 0;
    if (!read_exact_until(fd, reinterpret_cast<char*>(&network_length),
                          sizeof(network_length), deadline)) return std::nullopt;
    const auto length = ntohl(network_length);
    if (length > max_payload) throw std::runtime_error("payload length exceeds limit");
    std::string payload(length, '\0');
    if (!read_exact_until(fd, payload.data(), payload.size(), deadline))
        throw std::runtime_error("truncated frame");
    return payload;
}

// Helpers below are only the local test harness, not deadline-bounded send code.
struct Pair {
    int fd[2];
    Pair() {
        if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fd) < 0)
            throw std::system_error(errno, std::generic_category(), "socketpair");
    }
    ~Pair() { ::close(fd[0]); ::close(fd[1]); }
    Pair(const Pair&) = delete;
    Pair& operator=(const Pair&) = delete;
};

struct Join {
    std::thread thread;
    ~Join() { if (thread.joinable()) thread.join(); }
};

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

std::string pack(const std::string& body) {
    const auto length = htonl(static_cast<std::uint32_t>(body.size()));
    return std::string(reinterpret_cast<const char*>(&length), sizeof(length)) + body;
}

void send_small(int fd, const std::string& data) {
    std::size_t sent = 0;
    while (sent < data.size()) {
        const auto n = ::send(fd, data.data() + sent, data.size() - sent, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) throw std::runtime_error("test send failed");
        sent += static_cast<std::size_t>(n);
    }
}

template<class F> void rejects(F operation, const std::string& expected) {
    try { operation(); }
    catch (const std::exception& error) {
        require(std::string(error.what()).find(expected) != std::string::npos,
                "wrong failure category");
        return;
    }
    throw std::runtime_error("invalid input accepted");
}

int main() {
    try {
        {
            Pair pair;
            const std::string binary("C\0T", 3);
            send_small(pair.fd[0], pack(binary) + pack("") + pack("OK"));
            ::shutdown(pair.fd[0], SHUT_WR);
            const auto deadline = Clock::now() + 1s;
            require(read_frame_until(pair.fd[1], deadline) == binary, "binary frame");
            require(read_frame_until(pair.fd[1], deadline) == std::string(), "empty frame");
            require(read_frame_until(pair.fd[1], deadline) == "OK", "coalesced frame");
            require(!read_frame_until(pair.fd[1], deadline), "clean EOF");
        }
        for (const auto& broken : {pack("CAT").substr(0, 2), pack("CAT").substr(0, 4),
                                  pack("CAT").substr(0, 6)}) {
            Pair pair;
            send_small(pair.fd[0], broken);
            ::shutdown(pair.fd[0], SHUT_WR);
            rejects([&] { read_frame_until(pair.fd[1], Clock::now() + 1s); }, "truncated");
        }
        {
            Pair pair;
            const auto too_big = htonl(max_payload + 1);
            send_small(pair.fd[0], std::string(reinterpret_cast<const char*>(&too_big), 4));
            rejects([&] { read_frame_until(pair.fd[1], Clock::now() + 1s); }, "limit");
        }
        {
            Pair pair;
            const auto packet = pack("CAT");
            Join writer{std::thread([&] {
                for (const char byte : packet) {
                    ::send(pair.fd[0], &byte, 1, MSG_NOSIGNAL);
                    std::this_thread::sleep_for(2ms);
                }
            })};
            require(read_frame_until(pair.fd[1], Clock::now() + 1s) == "CAT", "fragmentation");
        }
        {
            Pair pair;
            const auto packet = pack("CAT");
            // Every gap is shorter than 100 ms, but the WHOLE frame takes >100 ms.
            Join writer{std::thread([&] {
                for (const char byte : packet) {
                    ::send(pair.fd[0], &byte, 1, MSG_NOSIGNAL);
                    std::this_thread::sleep_for(40ms);
                }
            })};
            rejects([&] { read_frame_until(pair.fd[1], Clock::now() + 100ms); }, "deadline");
        }
        {
            Pair pair;
            const auto packet = pack("CAT");
            Join writer{std::thread([&] {
                ::send(pair.fd[0], packet.data(), 3, MSG_NOSIGNAL);
                std::this_thread::sleep_for(60ms);
                ::send(pair.fd[0], packet.data() + 3, 1, MSG_NOSIGNAL);
                std::this_thread::sleep_for(60ms);
                ::send(pair.fd[0], packet.data() + 4, 3, MSG_NOSIGNAL);
            })};
            // Header and body each arrive within 100 ms of their own start;
            // together they exceed the one shared budget.
            rejects([&] { read_frame_until(pair.fd[1], Clock::now() + 100ms); }, "deadline");
        }
        {
            Pair pair;
            send_small(pair.fd[0], pack("ready"));
            rejects([&] { read_frame_until(pair.fd[1], Clock::now() - 1ms); }, "deadline");
        }
        std::cout << "PASS: binary/empty/coalesced/fragmented frames, clean/truncated EOF, "
                     "length limit, slow-drip/shared/expired deadlines\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
