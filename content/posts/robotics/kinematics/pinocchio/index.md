---
title: Pinocchio 机械臂逆运动学迭代数值解
date: 2024-08-18
lastmod: 2024-08-18
draft: false
tags: ["Robotics", "Kinematics", "Python"]
categories: ["编程技术"]
authors: ["chase"]
summary: "分享一个求解运动学逆解的第三方库 pinocchio， 并且根据其urdf文件中描述的关节极限范围内进行逆运动学求解的样例。"
showToc: true
TocOpen: true
hidemeta: false
comments: false
---


**https://github.com/stack-of-tasks/pinocchio**
分享一个求解运动学逆解的第三方库 pinocchio， 并且根据其urdf文件中描述的关节极限范围内进行逆运动学求解的样例。
```python
import numpy as np
from numpy.linalg import norm, solve
from IPython import embed
import pinocchio
from copy import deepcopy


class pinocchio_kinematics(object):

    def __init__(self, urdf_path: str):
        """Initializes the kinematics solver with a robot model."""
        self.urdf_path = urdf_path
        self.model = pinocchio.buildModelFromUrdf(self.urdf_path)
        self.model_data = self.model.createData()
        self.joint_id = self.model.getJointId("ee_link") - 1
        self.eps = 1e-4
        self.IT_MAX = 5000
        self.DT = 1e-1
        self.damp = 1e-12

    def qpos_to_limits(self, q: np.ndarray, upperPositionLimit: np.ndarray,
                       lowerPositionLimit: np.ndarray, joint_seed: np.ndarray,
                       ik_weight: np.ndarray):
        """Adjusts the joint positions (q) to be within specified limits and as close as possible to the joint seed,  
        while minimizing the total weighted difference.  
    
        Args:  
            q (np.ndarray): The original joint positions.  
            upperPositionLimit (np.ndarray): The upper limits for the joint positions.  
            lowerPositionLimit (np.ndarray): The lower limits for the joint positions.  
            joint_seed (np.ndarray): The desired (seed) joint positions.  
            ik_weight (np.ndarray): The weights to apply for each joint in the total difference calculation.  
    
        Returns:  
            np.ndarray: The adjusted joint positions within the specified limits.  
        """
        qpos_limit = np.copy(q)
        best_qpos_limit = np.copy(q)
        best_total_q_diff = float('inf')

        if ik_weight is None:
            ik_weight = np.ones_like(q)

        for i in range(len(q)):
            # Generate multiple candidates by adding or subtracting 2*pi multiples
            candidates = []
            for k in range(
                    -5, 6
            ):  # You can adjust the range of k to explore more possibilities
                candidate = q[i] + k * 2 * np.pi
                if lowerPositionLimit[i] <= candidate <= upperPositionLimit[i]:
                    candidates.append(candidate)

            # If no candidates are within the limits, just use the original value adjusted with 2*pi multiples
            if not candidates:
                candidate = (q[i] - joint_seed[i]) % (2 * np.pi) + joint_seed[i]
                while candidate < lowerPositionLimit[i]:
                    candidate += 2 * np.pi
                while candidate > upperPositionLimit[i]:
                    candidate -= 2 * np.pi
                candidates.append(candidate)

            # Find the candidate that gives the smallest total_q_diff
            best_candidate_diff = float('inf')
            best_candidate = candidates[0]
            for candidate in candidates:
                qpos_limit[i] = candidate
                total_q_diff = np.sum(
                    np.abs(qpos_limit - joint_seed) * ik_weight)
                if total_q_diff < best_candidate_diff:
                    best_candidate_diff = total_q_diff
                    best_candidate = candidate

            qpos_limit[i] = best_candidate
            if best_candidate_diff < best_total_q_diff:
                best_total_q_diff = best_candidate_diff
                best_qpos_limit = np.copy(qpos_limit)

        return best_qpos_limit

    def get_ik(self,
               target_pose: np.ndarray,
               joint_seed: np.ndarray,
               ik_weight: np.ndarray = None):
        """Computes the inverse kinematics for a given target pose."""
        if not isinstance(target_pose,
                          np.ndarray) or target_pose.shape != (4, 4):
            raise ValueError("target_pose must be a 4x4 numpy array")
        if not isinstance(joint_seed, np.ndarray):
            raise ValueError("joint_seed must be of type np.ndarray")
        target_pose_SE3 = pinocchio.SE3(target_pose)
        q = deepcopy(joint_seed).astype(np.float64)

        for i in range(self.IT_MAX):
            pinocchio.forwardKinematics(self.model, self.model_data, q)
            dMi = target_pose_SE3.actInv(self.model_data.oMi[self.joint_id])
            err = pinocchio.log(dMi).vector
            if norm(err) < self.eps:
                print("Pin:Convergence achieved!")
                
                q = self.qpos_to_limits(q, self.model.upperPositionLimit,
                                        self.model.lowerPositionLimit,
                                        joint_seed, ik_weight)
                return True, q

            J = pinocchio.computeJointJacobian(self.model, self.model_data, q,
                                               self.joint_id)
            v = -J.T.dot(solve(J.dot(J.T) + self.damp * np.eye(6), err))
            q = pinocchio.integrate(self.model, q, v * self.DT)
            if not i % 10:
                print("Pin:{} error = {}!".format(i, err.T))

        print(
            "Pin:The iterative algorithm has not reached convergence to the desired precision"
        )
        return False, q


if __name__ == "__main__":
    np.set_printoptions(4, suppress=True)
	urdf_path = "../Agile/Diana_7/Diana_7.urdf"
    kin_solver = pinocchio_kinematics(urdf_path)

    pose1 = np.array([[-0., -1., 0., 0.2465], [-1., -0., -0., -0.1029],
                      [0., 0., -1., 0.9301], [0., 0., 0., 1.]])
    joint_seed = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    res, joint = kin_solver.get_ik(pose1, joint_seed)

    embed()

```
