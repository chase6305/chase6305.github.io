#!/usr/bin/env python3
"""Run article snippets with disposable fixtures, never robot hardware.

Qt uses offscreen; MeshCat records serialized commands without starting a server.
tmux uses a private socket and stub watch, never the user's existing sessions.
"""
import argparse
import contextlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
from validate_blog import ROOT, Page, code_blocks, check_structured_snippet, parse_jsonc


def blocks(post, language="python"):
    return [code for lang, code, _ in code_blocks(
        (ROOT / "content/posts" / post).read_text()) if lang == language]


def run(source, values=None):
    values = {"__name__": "blog_test"} if values is None else values
    exec(compile(source, "<article>", "exec"), values)
    return values


def expect_error(kind, function, *args):
    try:
        function(*args)
    except kind:
        return
    raise AssertionError(f"Expected {kind.__name__}")


def core():
    formatting = blocks("python/f_string/index.md")
    for source in formatting:
        run(source)
    unicode_blocks = blocks("unicode/index.md")
    for source in unicode_blocks:
        ns = run(source)
        if "format_status" in ns:
            expect_error(ValueError, ns["format_status"], "ready", "unknown")
            failure = RuntimeError("task failed")
            def fail(_):
                raise failure
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                try:
                    ns["process_task"](1, fail, False)
                except RuntimeError as caught:
                    assert caught is failure
                else:
                    raise AssertionError("Task failure was swallowed")
            assert output.getvalue().startswith("[ERROR]")
            assert ns["format_status"]("a\rb\nc", "error", False) == "[ERROR] a\\rb\\nc"
    import codecs
    expect_error(UnicodeDecodeError, codecs.getincrementaldecoder("utf-8")().decode,
                 b"\xe4", True)
    standards = []
    with tempfile.TemporaryDirectory(prefix="chase-round4-core-") as folder:
        directory = Path(folder)
        source = directory / "unicode.cpp"
        source.write_text(blocks("unicode/index.md", "cpp")[0])
        for standard in ("c++17", "c++20"):
            binary = directory / standard
            subprocess.run(["g++", "-std=" + standard, "-Wall", "-Wextra", "-Werror",
                            str(source), "-o", str(binary)], check=True, timeout=30)
            subprocess.run([str(binary)], check=True, timeout=5)
            standards.append(standard)
        marker = directory / "must-not-exist"
        check_structured_snippet("bash", "touch " + shlex.quote(str(marker)))
        assert not marker.exists(), "Syntax validation executed a command"
    for language, source in [("bash", "if true; then"), ("json", '{"x":}'),
                             ("jsonc", '{"x": /* broken'), ("xml", "<a>")]:
        try:
            check_structured_snippet(language, source)
        except Exception:
            pass
        else:
            raise AssertionError(f"Invalid {language} accepted")
    assert parse_jsonc('{"url":"https://example.org/a//b",/*comment*/"x":[1,],}') == {
        "url": "https://example.org/a//b", "x": [1]}
    assert parse_jsonc('{"literal":",} // /* text */",}')["literal"] == ",} // /* text */"
    escaped = "\\`" * 3
    expect_error(ValueError, list, code_blocks(escaped + "python\nx=1\n" + escaped))
    expect_error(ValueError, list, code_blocks("```python\nx=1"))
    assert list(code_blocks("  ~~~python\n  x = 1\n  ~~~")) == [("python", "x = 1\n", 1)]
    assert Page('<a id="same"></a><h2 id="same"></h2>').duplicate_ids == {"same"}
    return {"f_string_blocks": len(formatting), "unicode_blocks": len(unicode_blocks),
            "unicode_cpp": standards, "exception_propagation": "passed",
            "validator_regression": "fences, JSONC, shell non-execution, XML, duplicate ids passed"}


def jacobian():
    import numpy as np
    import pinocchio as pin
    source = blocks("robotics/kinematics/jacobian/index.md")
    ns = run(source[0])
    fixture = ROOT / "content/posts/robotics/kinematics/pytorch/planar2r.urdf"
    model, data, q, frame_id, J = ns["load_robot"](fixture, "tip")
    ns.update(model=model, data=data, q=q, frame_id=frame_id, J=J)
    expect_error(ValueError, ns["load_robot"], fixture, "missing_frame")
    for snippet in source[1:]:
        run(snippet, ns)
    np.testing.assert_allclose(ns["xi_from_jacobian"], ns["xi_from_pinocchio"], atol=1e-12)
    assert np.isfinite(ns["qdot"]).all()
    rng = np.random.default_rng(4)
    worst_position = worst_time = 0.0
    for _ in range(20):
        q = rng.uniform(-1.5, 1.5, model.nq)
        velocity = rng.normal(size=model.nv)
        pin.forwardKinematics(model, data, q, velocity)
        pin.updateFramePlacements(model, data)
        pose = data.oMf[frame_id].copy()
        all_j = {}
        for frame in (pin.LOCAL, pin.LOCAL_WORLD_ALIGNED, pin.WORLD):
            jac = pin.computeFrameJacobian(model, data, q, frame_id, frame).copy()
            pin.forwardKinematics(model, data, q, velocity)
            pin.updateFramePlacements(model, data)
            np.testing.assert_allclose(jac @ velocity,
                pin.getFrameVelocity(model, data, frame_id, frame).vector, atol=1e-12)
            all_j[frame] = jac
            eps = 1e-7
            plus = pin.computeFrameJacobian(model, data,
                pin.integrate(model, q, eps * velocity), frame_id, frame).copy()
            minus = pin.computeFrameJacobian(model, data,
                pin.integrate(model, q, -eps * velocity), frame_id, frame).copy()
            pin.computeJointJacobiansTimeVariation(model, data, q, velocity)
            pin.updateFramePlacements(model, data)
            exact = pin.getFrameJacobianTimeVariation(model, data, frame_id, frame).copy()
            numeric = (plus - minus) / (2 * eps)
            np.testing.assert_allclose(exact, numeric, atol=2e-8, rtol=1e-6)
            worst_time = max(worst_time, float(np.max(np.abs(exact - numeric))))
        rotate = np.zeros((6, 6))
        rotate[:3, :3] = rotate[3:, 3:] = pose.rotation
        np.testing.assert_allclose(all_j[pin.WORLD], pose.action @ all_j[pin.LOCAL], atol=1e-12)
        np.testing.assert_allclose(all_j[pin.LOCAL_WORLD_ALIGNED],
                                   rotate @ all_j[pin.LOCAL], atol=1e-12)
        ns.update(q=q, J=all_j[pin.LOCAL_WORLD_ALIGNED])
        run(next(s for s in source if "J_pos_fd =" in s), ns)
        worst_position = max(worst_position, float(np.max(np.abs(ns["J_pos_fd"] - ns["J"][:3]))))
    J = rng.normal(size=(6, 7))
    N = np.eye(7) - np.linalg.pinv(J) @ J
    np.testing.assert_allclose(J @ N, 0, atol=1e-12)
    assert np.linalg.matrix_rank(N, tol=1e-10) == 1
    damped = J.T @ np.linalg.solve(J @ J.T + .1**2 * np.eye(6), np.eye(6))
    leakage = float(np.linalg.norm(J @ (np.eye(7) - damped @ J)))
    assert leakage > 1e-5
    return {"pinocchio": pin.__version__, "article_blocks": len(source),
            "random_poses": 20, "reference_frames": 3,
            "position_fd_max_error": worst_position, "jacobian_time_fd_max_error": worst_time,
            "damped_nullspace_leakage_norm": leakage}


def qt():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    compiler = shutil.which("pyside6-uic") or str(Path(sys.executable).with_name("pyside6-uic"))
    with tempfile.TemporaryDirectory(prefix="chase-round4-qt-") as folder:
        directory = Path(folder)
        ui = directory / "UI"
        ui.mkdir()
        (ui / "__init__.py").write_text("")
        fixture = ROOT / "content/posts/gui/qt/pyside6/robot-dialog.ui"
        shutil.copyfile(fixture, ui / "robot-dialog.ui")
        def generate():
            subprocess.run([compiler, str(ui / "robot-dialog.ui"), "-o",
                            str(ui / "robot_dialog.py")], check=True, timeout=30)
        generate()
        sys.path.insert(0, str(directory))
        try:
            source = blocks("gui/qt/pyside6/index.md")[0]
            window = run(source)["RobotDialog"]()
            window.ui.applyButton.click()
            window.ui.applyButton.click()
            assert window.applied_count == 2
            assert window.ui.statusLabel.text() == "已应用 2 次"
            window.close()
            (ui / "robot-dialog.ui").write_text(fixture.read_text().replace("机器人设置", "重新生成界面"))
            generate()
            (directory / "main.py").write_text(source)
            subprocess.run([sys.executable, "-B", "-c",
                "from PySide6.QtWidgets import QApplication; from main import RobotDialog; "
                "app=QApplication([]); w=RobotDialog(); w.ui.applyButton.click(); "
                "assert w.windowTitle()=='重新生成界面'; "
                "assert w.ui.statusLabel.text()=='已应用 1 次'; w.close()"],
                check=True, timeout=30, cwd=directory,
                env=dict(os.environ, PYTHONPYCACHEPREFIX=str(directory / "fresh-cache")))
        finally:
            sys.path.pop(0)
        previous = Path.cwd()
        try:
            os.chdir(directory)
            run(blocks("qt/pyside6_matplotlib/index.md")[0])
            assert (directory / "output.png").stat().st_size > 1000
        finally:
            os.chdir(previous)
    PlotWindow = run(blocks("qt/pyside6_matplotlib/index.md")[1])["PlotWindow"]
    for _ in range(3):
        plot = PlotWindow()
        plot.show()
        for _ in range(50):
            plot.update_plot()
        assert len(plot.axes.lines) == 1
        phase = plot.phase
        loop = QEventLoop()
        QTimer.singleShot(250, loop.quit)
        loop.exec()
        assert plot.phase > phase, "Timer failed to update the plot"
        plot.canvas.draw()
        plot.close()
        assert not plot.timer.isActive()
        plot.deleteLater()
        app.processEvents()
    return {"pyside6": version("PySide6"), "matplotlib": version("matplotlib"),
            "ui_regeneration_and_signals": "passed", "plot_windows": 3,
            "manual_updates_per_window": 50, "timer_events_and_close": "passed",
            "display_mode": "offscreen; not a native desktop display test"}


def meshcat():
    import meshcat
    import numpy as np
    import umsgpack
    class RecordingWindow:
        web_url = "recording://offline"
        def __init__(self):
            self.commands = []
        def send(self, command):
            payload = command.lower()
            umsgpack.packb(payload)
            self.commands.append(payload)
    window = RecordingWindow()
    source = blocks("meshcat/index.md")
    assert source[0].count("meshcat.Visualizer()") == 1
    ns = run(source[0].replace("meshcat.Visualizer()", "meshcat.Visualizer(window=window)"),
             {"__name__": "blog_test", "window": window, "input": lambda _: ""})
    cylinder = next(c for c in window.commands if c["type"] == "set_transform"
                    and c["path"].endswith("/cylinder"))
    transform = np.array(cylinder["matrix"]).reshape(4, 4, order="F")
    endpoints = [transform @ [0, y, 0, 1] for y in (-.3, .3)]
    np.testing.assert_allclose(np.array(endpoints)[:, 2], [0, .6], atol=1e-12)
    assert sum(c["type"] == "set_object" for c in window.commands) == 4
    run(source[1], ns)
    assert window.commands[-1] == {"type": "delete", "path": "/meshcat/demo/tetrahedron"}
    return {"meshcat": version("meshcat"), "serialized_commands": len(window.commands),
            "cylinder_z_endpoints": [float(point[2]) for point in endpoints],
            "transport": "recording sink; no HTTP, WebSocket or ZeroMQ server started"}


def tmux(binary):
    binary = shutil.which(binary) or str(Path(binary).resolve())
    with tempfile.TemporaryDirectory(prefix="chase-round4-tmux-") as folder:
        directory = Path(folder)
        socket = directory / "private.sock"
        shim = directory / "bin"
        shim.mkdir()
        (shim / "tmux").write_text("#!/bin/sh\nexec " + shlex.quote(binary) +
            " -S " + shlex.quote(str(socket)) + ' -f /dev/null "$@"\n')
        (shim / "watch").write_text("#!/bin/sh\nexec sleep 30\n")
        for path in shim.iterdir():
            path.chmod(0o755)
        env = dict(os.environ, PATH=str(shim) + os.pathsep + os.environ["PATH"],
                   TMUX="", BASH_ENV="", ENV="")
        def command(*args, check=True):
            return subprocess.run([str(shim / "tmux"), *args], check=check, env=env,
                                  text=True, capture_output=True, timeout=10)
        script = next(c for c in blocks("linux/watch/index.md", "bash")
                      if "monitor_session=blog-monitor" in c)
        try:
            command("new-session", "-d", "-s", "spectator", "sleep 30")
            before = command("list-panes", "-s", "-t", "=spectator", "-F", "#{pane_id}").stdout
            first = subprocess.run(["bash"], input=script, text=True, env=env,
                                   capture_output=True, timeout=10)
            assert first.returncode == 0, first.stderr
            panes = command("list-panes", "-s", "-t", "=blog-monitor", "-F", "#{pane_id}").stdout.splitlines()
            assert len(panes) == 3 and len(set(panes)) == 3
            after = command("list-panes", "-s", "-t", "=spectator", "-F", "#{pane_id}").stdout
            assert before == after, "An unrelated session changed"
            second = subprocess.run(["bash"], input=script, text=True, env=env,
                                    capture_output=True, timeout=10)
            assert second.returncode == 1 and "已存在" in second.stderr
            assert command("list-panes", "-s", "-t", "=blog-monitor", "-F", "#{pane_id}").stdout.splitlines() == panes
        finally:
            command("kill-server", check=False)  # Only this disposable socket.
    return {"version": subprocess.check_output([binary, "-V"], text=True).strip(),
            "new_panes": 3, "unrelated_session": "unchanged",
            "repeat_execution": "refused without changes", "watch": "stubbed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", choices=["core", "jacobian", "qt", "meshcat", "tmux"],
                        default=["core"])
    parser.add_argument("--tmux-binary", default="tmux")
    args = parser.parse_args()
    result = {}
    for group in args.groups:
        result[group] = tmux(args.tmux_binary) if group == "tmux" else globals()[group]()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
