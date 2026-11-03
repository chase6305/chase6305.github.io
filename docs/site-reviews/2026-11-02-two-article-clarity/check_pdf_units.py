"""Check selected short explanatory units in actual native PDF text pages."""
import argparse
import json
from pathlib import Path


def compact(text):
    return "".join(text.split()).replace("\u200b", "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    units = {
        "supermimic": {
            "three_time_windows": [
                "每20ms一次控制输出", "8个控制步，代表0.16s轨迹", "32个控制步，代表0.64s轨迹",
            ],
            "phase_parameter_comparison": [
                "只更新回放数据", "Actorθ；模型与critic参数固定", "Criticψ；样本与监督目标固定",
            ],
            "torch_code_and_output": [
                "import torch", 'print(f"model gradient={gain.grad}")',
                "输出为 actor gradient=8.0 和 model gradient=None。",
            ],
            "three_gradient_presets": [
                "完整 BPTT 8 −0.488", "状态 detach 5.6 −0.61472", "反向模型 −8 −1.768",
            ],
            "critic_definition_and_rows": [
                "表中的窗口回报已经应用折扣与存活权重", "A 0.50 0.10 0.55", "B 0.48 0.20 0.58",
            ],
        },
        "rlhf-rlvr-rlcd": {
            "three_constructed_candidates": [
                "正确解释None表示", "但声称已经处理零除数", "却声称除数为零时返回数值0",
            ],
            "model_role_comparison": [
                "下一个token生成什么", "这批回答采集时的概率是多少",
            ],
            "three_feedback_protocols": [
                "本例选择A", "不检查解释文字", "按生成分支自动标记",
            ],
            "preference_difference_and_rows": [
                "沿用 A 优于 C 的记录", "0 0.500000 0.693147", "2 0.880797 0.126928",
            ],
        },
    }
    rows = []
    files = sorted(args.directory.glob("*.txt"))
    if not files:
        raise SystemExit("No exported native PDF text files")
    for file in files:
        article = "supermimic" if file.name.startswith("supermimic-") else "rlhf-rlvr-rlcd"
        pages = [compact(p) for p in file.read_text().split("\f")]
        for name, tokens in units[article].items():
            matches = [n for n, page in enumerate(pages, 1)
                       if all(compact(token) in page for token in tokens)]
            rows.append({"file": file.with_suffix(".pdf").name, "unit": name,
                         "tokens": tokens, "pages": matches, "passed": len(matches) == 1})
    report = {"pdfs": len(files), "cases": len(rows), "failures": sum(not r["passed"] for r in rows),
              "scope": "Nine selected short units stay on one native PDF page. This does not replace visual review or audit every article table.",
              "rows": rows}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("pdfs", "cases", "failures")}))
    if report["failures"]:
        raise SystemExit("Inspect split or missing explanatory units")


if __name__ == "__main__":
    main()
