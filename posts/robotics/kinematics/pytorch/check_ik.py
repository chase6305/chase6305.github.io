import argparse
from pathlib import Path

import torch
import pytorch_kinematics as pk

parser = argparse.ArgumentParser()
parser.add_argument("urdf")
parser.add_argument("end_link")
args = parser.parse_args()

torch.manual_seed(42)
device = torch.device("cpu")  # 先在 CPU 验证，再比较 CUDA 批量性能
dtype = torch.float64
chain = pk.build_serial_chain_from_urdf(
    Path(args.urdf).read_text(encoding="utf-8"), args.end_link
).to(dtype=dtype, device=device)
names = chain.get_joint_parameter_names()
limits = torch.tensor(chain.get_joint_limits(), dtype=dtype, device=device).T
if limits.shape != (len(names), 2):
    raise ValueError("Unexpected joint-limit shape")
if not torch.isfinite(limits).all() or not (limits[:, 0] < limits[:, 1]).all():
    raise ValueError("This example requires finite, ordered joint limits")

lower, upper = limits[:, 0], limits[:, 1]
q_known = (lower + 0.45 * (upper - lower)).unsqueeze(0)
target = chain.forward_kinematics(q_known)
target_matrix = target.get_matrix().detach()

solver = pk.PseudoInverseIK(
    chain,
    joint_limits=limits,
    num_retries=20,
    max_iterations=200,
    pos_tolerance=1e-4,
    rot_tolerance=1e-4,
    early_stopping_any_converged=True,
    lr=0.2,
)
result = solver.solve(target)
candidates = result.solutions[0][result.converged[0]]
if candidates.numel() == 0:
    raise RuntimeError("No converged retry; inspect residuals and initializations")
valid = (
    torch.isfinite(candidates).all(dim=-1)
    & (candidates >= lower - 1e-8).all(dim=-1)
    & (candidates <= upper + 1e-8).all(dim=-1)
)
candidates = candidates[valid]
if candidates.numel() == 0:
    raise RuntimeError("Converged retries failed joint-limit validation")

# 在有限关节区间内选最接近参考姿态的一组；连续关节需另做周期距离。
distances = torch.linalg.vector_norm(candidates - q_known, dim=-1)
q_solution = candidates[distances.argmin()].unsqueeze(0)
actual = chain.forward_kinematics(q_solution).get_matrix()
position_error = torch.linalg.vector_norm(
    actual[0, :3, 3] - target_matrix[0, :3, 3]
)
relative_rotation = actual[0, :3, :3].T @ target_matrix[0, :3, :3]
cosine = ((torch.trace(relative_rotation) - 1) / 2).clamp(-1, 1)
rotation_error = torch.acos(cosine)
print("joints:", names)
print("q:", q_solution)
print("position error (m):", position_error.item())
print("rotation error (rad):", rotation_error.item())
if position_error > 2e-4 or rotation_error > 2e-4:
    raise RuntimeError("Independent FK check failed")
