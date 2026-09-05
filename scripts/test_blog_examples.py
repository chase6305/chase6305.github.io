#!/usr/bin/env python3
"""Regression checks extracted from the articles; no hardware or GPU access.

Requires Python 3.10+, NumPy, Pinocchio, g++, CMake. Network tests bind only
loopback, using ephemeral ports substituted into the extracted demonstrations.
All compiled output lives in a TemporaryDirectory, never in the repository.
"""
import argparse
import contextlib
import io
import json
import selectors
import socket
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pinocchio as pin

from validate_blog import ROOT, code_blocks


def blocks(post, language):
    text = (ROOT / "content/posts" / post).read_text(encoding="utf-8")
    return [code for lang, code, _ in code_blocks(text) if lang == language]


def namespace(code):
    result = {"__name__": "blog_example"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(code, "<blog-example>", "exec"), result)
    return result


def run(command, **kwargs):
    result = subprocess.run(command, text=True, capture_output=True,
                            timeout=40, **kwargs)
    if result.returncode:
        raise AssertionError(f"{command}: {result.stdout}\n{result.stderr}")
    return result.stdout.strip()


def compile_cpp(directory, name, source):
    target = directory / name
    run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic",
         "-pthread", "-x", "c++", "-", "-o", str(target)], input=source)
    return str(target)


def test_cpp(directory):
    results = {}
    for name, post in [
        ("astar", "ai/algorithms/Astar/Astar-cpp/index.md"),
        ("spsc", "queue/index.md"),
        ("ownership", "cpp/smart-pointer/index.md"),
    ]:
        source = blocks(post, "cpp")[0]
        binary = compile_cpp(directory, name, source)
        results[name] = run([binary])
        if name == "astar":
            extra = source.replace("int main()", "int article_main()")
            # A renamed main needs an explicit return to avoid undefined behavior.
            extra = extra.replace('<< path_length(guided) << \'\\n\';',
                                  '<< path_length(guided) << \'\\n\'; return 0;')
            extra += """
#include <random>
int main() {
    article_main();
    std::mt19937 rng(42);
    for (int trial=0; trial<200; ++trial) {
        Grid grid({6,6,3});
        for (auto& cell : grid.occupied) cell = (rng() % 100 < 25);
        Cell start{0,0,0}, goal{5,5,2};
        grid.occupied[grid.id(start)] = grid.occupied[grid.id(goal)] = 0;
        for (bool diagonal : {false, true}) {
            auto a = astar(grid, start, goal, diagonal);
            auto d = astar(grid, start, goal, diagonal, false);
            assert(a.empty() == d.empty());
            assert(std::abs(path_length(a)-path_length(d)) < 1e-9);
            for (std::size_t i=1; i<a.size(); ++i) {
                Cell delta{};
                for (int j=0; j<3; ++j) delta[j] = a[i][j]-a[i-1][j];
                assert(clear_step(grid, a[i-1], delta));
            }
        }
    }
}
"""
            run([compile_cpp(directory, "astar_random", extra)])
            results["astar_random_cases"] = 400
    std_programs = [code for code in blocks("cpp/std/index.md", "cpp")
                    if "int main(" in code]
    for index, source in enumerate(std_programs):
        run([compile_cpp(directory, f"std_{index}", source)])
    results["std_programs"] = len(std_programs)
    return results


def test_cmake(directory):
    root = directory / "cmake_example"
    external = root / "external/hello"
    external.mkdir(parents=True)
    cmake = blocks("cmake/ExternalProject_Add/index.md", "cmake")
    cpp = blocks("cmake/ExternalProject_Add/index.md", "cpp")
    for path, source in [(root / "CMakeLists.txt", cmake[0]),
                         (external / "CMakeLists.txt", cmake[1]),
                         (external / "hello.cpp", cpp[0]),
                         (root / "main.cpp", cpp[1])]:
        path.write_text(source, encoding="utf-8")
    build = root / "build"
    run(["cmake", "-S", str(root), "-B", str(build)])
    for _ in range(2):
        run(["cmake", "--build", str(build), "--parallel", "2"])
        assert run([str(build / "app")]) == "42"
    return "clean and incremental builds passed"


def test_python():
    result = {}
    pid = namespace(blocks("pid/index.md", "python")[0])["PID"]
    controller = pid(2.0, 1.0, 0.1, 3.0)
    for _ in range(1000):
        assert controller.update(100.0, 0.0, 0.01) == 3.0
    assert controller.integral == 0.0
    for bad_dt in (0, -1, float("nan")):
        try:
            controller.update(1, 0, bad_dt)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid PID dt accepted")
    result["pid"] = "saturation, anti-windup, invalid dt passed"

    rl = blocks("rl/index.md", "python")
    cliff = namespace(next(code for code in rl if "def train(method" in code))
    assert cliff["step"](36, 1) == (36, -100.0, False)
    assert cliff["step"](35, 2) == (47, -1.0, True)
    result["cliff_last_100_mean"] = {
        method: float(cliff["train"](method)[1][-100:].mean())
        for method in ("sarsa", "q_learning")
    }
    advantage = namespace(next(code for code in rl
                               if "def n_step_advantage" in code))["n_step_advantage"]
    assert abs(advantage([1, 2], [0, 0, 3], [False, False], 0, 5, .9)
               - (1 + .9*2 + .9**2*3)) < 1e-12
    assert abs(advantage([1, 2], [0, 0, 3], [False, True], 0, 5, .9)
               - (1 + .9*2)) < 1e-12
    result["n_step_advantage"] = "termination and truncation passed"

    srs = namespace(blocks(
        "robotics/kinematics/seven-dof-kinematics/index.md", "python")[0])["SRSKinSolver"]()
    rng = np.random.default_rng(42)
    largest = 0.0
    for _ in range(100):
        known = rng.uniform(-2, 2, 7)
        target, _ = srs.compute_total_transform(known)
        for branch in range(8):
            solution, _, _ = srs.inverse_kinematics(target, rng.uniform(-np.pi, np.pi), branch)
            actual, _ = srs.compute_total_transform(solution)
            largest = max(largest, float(np.abs(actual-target).max()))
            assert np.allclose(actual, target, atol=1e-8)
    result["srs"] = {"cases": 800, "max_transform_error": largest}

    solve = namespace(blocks("robotics/kinematics/pinocchio/index.md", "python")[0])["solve_ik"]
    model = pin.buildSampleModelManipulator()
    model.lowerPositionLimit[:] = -np.pi
    model.upperPositionLimit[:] = np.pi
    frame = model.frames[-1].name
    fid = model.getFrameId(frame)
    data = model.createData()
    largest = 0.0
    for _ in range(30):
        known = rng.uniform(-.7, .7, model.nq)
        pin.forwardKinematics(model, data, known)
        pin.updateFramePlacements(model, data)
        target = data.oMf[fid].copy()
        seed = known + rng.uniform(-.05, .05, model.nq)
        solved = solve(model, frame, target, seed)
        assert solved["success"], solved
        largest = max(largest, solved["position_error"], solved["rotation_error"])
    failed = solve(model, frame, pin.SE3(np.eye(3), np.array([100., 0., 0.])),
                   pin.neutral(model), max_iter=5)
    assert not failed["success"]
    result["pinocchio"] = {"reachable_cases": 30, "unreachable_reported": True,
                          "max_residual": largest, "version": pin.__version__}
    return result


def test_modbus():
    ns = namespace(blocks("dialout/dh/index.md", "python")[0])
    crc, with_crc, bus_type = ns["crc16"], ns["with_crc"], ns["ModbusRegisterBus"]
    assert crc(bytes.fromhex("01 03 00 00 00 0a")) == 0xCDC5

    class Port:
        def __init__(self, response=None):
            self.response, self.pending, self.writes = response, b"", []

        def write(self, data):
            assert not self.pending, "Overlapping transactions"
            self.writes.append(data)
            self.pending = self.response if self.response is not None else data
            return len(data)

        def read(self, length):
            result, self.pending = self.pending[:1], self.pending[1:]
            return result

        def close(self):
            pass

    response = with_crc(bytes.fromhex("01 03 02 12 34"))
    port = Port(response)
    bus = bus_type(port, frame_gap=.0001)
    assert bus.read_register(0) == 0x1234
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(bus.read_register, [0, 1])) == [0x1234, 0x1234]
    bus_type(Port()).write_register(0x100, 42)
    bad = [response[:-1] + bytes([response[-1] ^ 1]),
           with_crc(bytes.fromhex("02 03 02 12 34")),
           with_crc(bytes.fromhex("01 83 02")), b"\x01"]
    for response in bad:
        failed_bus = bus_type(Port(response), timeout=.002, frame_gap=.0001)
        try:
            failed_bus.read_register(0)
        except (ValueError, RuntimeError, TimeoutError):
            assert failed_bus.failed
        else:
            raise AssertionError("Invalid response accepted")
    return "CRC, byte-wise reads, write echo, concurrency, faults and timeout passed"


def free_port(kind):
    with socket.socket(socket.AF_INET, kind) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@contextlib.contextmanager
def server(binary):
    child = subprocess.Popen([binary, "server"], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    try:
        with selectors.DefaultSelector() as ready:
            ready.register(child.stdout, selectors.EVENT_READ)
            assert ready.select(timeout=5), 'Server did not become ready'
        assert "Listening" in child.stdout.readline(), child.stderr.read()
        yield child
    finally:
        if child.poll() is None:
            child.terminate()
        child.communicate(timeout=5)


def test_network(directory):
    tcp_port, udp_port = free_port(socket.SOCK_STREAM), free_port(socket.SOCK_DGRAM)
    tcp = compile_cpp(directory, "tcp",
                      blocks("network-protocol/c++_tcp.md", "cpp")[0]
                      .replace("htons(8888)", f"htons({tcp_port})"))
    udp = compile_cpp(directory, "udp",
                      blocks("network-protocol/c++_udp.md", "cpp")[0]
                      .replace("htons(5001)", f"htons({udp_port})"))
    with server(tcp) as child:
        assert "Hello, framed TCP!" in run([tcp, "client"])
        assert child.wait(timeout=5) == 0
    with server(tcp) as child, socket.create_connection(("127.0.0.1", tcp_port)) as peer:
        peer.settimeout(3)
        body = b"fragmented\0payload"
        packet = struct.pack("!I", len(body)) + body
        for value in packet:
            peer.sendall(bytes([value]))
        received = b""
        while len(received) < len(packet):
            received += peer.recv(len(packet)-len(received))
        assert received == packet
        assert child.wait(timeout=5) == 0
    with server(udp) as child:
        assert "Hello, UDP!" in run([udp, "client"])
        assert child.wait(timeout=5) == 0
    with server(udp) as child, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as peer:
        peer.settimeout(3)
        peer.sendto(b"", ("127.0.0.1", udp_port))
        assert peer.recvfrom(1500)[0] == b""
        assert child.wait(timeout=5) == 0
    with server(udp) as child, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as peer:
        peer.sendto(b"x" * 1600, ("127.0.0.1", udp_port))
        assert child.wait(timeout=5) == 1
    return "TCP echo/fragmentation; UDP echo/empty/truncated datagrams passed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", action="store_true",
                        help="also run loopback-only socket tests")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="chase-blog-tests-") as name:
        directory = Path(name)
        results = {"cpp": test_cpp(directory), "cmake": test_cmake(directory),
                   "python": test_python(), "modbus": test_modbus()}
        if args.network:
            results["network"] = test_network(directory)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
