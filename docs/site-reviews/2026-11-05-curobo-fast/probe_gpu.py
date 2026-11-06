#!/usr/bin/env python3
"""One pinned-source Franka GPU smoke query; never connects to hardware.

Use a prepared CUDA environment. Does not install packages or change the source
checkout. Compile caches and results are isolated under --output. A successful
single query is not a timing/accuracy reproduction of the paper benchmark.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REVISION='b29a8c2ee6869585bb05b056b9ab53b9c0753106'

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.source_root.resolve();root=args.output.resolve()
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if revision!=REVISION:raise ValueError(f'Expected {REVISION}, found {revision}')
    root.mkdir(parents=True,exist_ok=False)
    os.environ.update(XDG_CACHE_HOME=str(root/'cache'),CUDA_CACHE_PATH=str(root/'cuda-cache'),
                      TORCH_EXTENSIONS_DIR=str(root/'torch-extensions'),PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(source))
    import torch
    import warp as wp
    wp.config.kernel_cache_dir=str(root/'warp-cache')
    import curobo
    import curobo.runtime as runtime
    import curobo._src.runtime as internal_runtime
    runtime.cache_dir=internal_runtime.cache_dir=str(root/'curobo-cache')
    from curobo.motion_planner import MotionPlanner,MotionPlannerCfg
    from curobo.types import JointState
    if not torch.cuda.is_available():raise RuntimeError('A working CUDA GPU is required')
    config=MotionPlannerCfg.create(robot='franka.yml',scene_model='collision_test.yml',
                                   graph_planner_algorithm='rrt',enable_timeopt=True)
    with MotionPlanner(config) as planner:
        planner.warmup(enable_graph=True,num_warmup_iterations=2)
        start=JointState.from_position(planner.default_joint_state.position.unsqueeze(0),
                                       joint_names=planner.joint_names)
        goal=start.clone();goal.position[:,0]+=.2
        target=planner.compute_kinematics(goal).tool_poses.as_goal()
        torch.cuda.synchronize();before=time.perf_counter()
        result=planner.plan_pose(target,start,max_attempts=1,enable_graph_attempt=0)
        torch.cuda.synchronize();elapsed=time.perf_counter()-before
        assert result is not None and result.success is not None and bool(result.success.all())
        trajectory=result.get_interpolated_plan()
        assert trajectory.joint_names is not None
        assert len(trajectory.joint_names)==trajectory.position.shape[-1]
        assert bool(torch.isfinite(trajectory.position).all())
        report=dict(source_revision=revision,source_file=curobo.__file__,
                    gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                    robot='franka.yml',scene='collision_test.yml',
                    graph_planner_algorithm='rrt',enable_timeopt=True,
                    cuda_graphs=internal_runtime.cuda_graphs,success=result.success.cpu().tolist(),
                    trajectory_source=result.trajectory_source,shape=list(trajectory.position.shape),
                    actuated_joint_names=planner.joint_names,trajectory_joint_names=trajectory.joint_names,
                    trajectory_dt=None if trajectory.dt is None else trajectory.dt.cpu().tolist(),
                    solver_reported_seconds=result.solve_time,synchronized_single_call_seconds=elapsed,
                    note='One smoke query, possible first-layout setup overhead; not a benchmark or physical-robot validation')
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
