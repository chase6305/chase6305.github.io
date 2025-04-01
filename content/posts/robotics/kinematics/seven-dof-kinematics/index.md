---
title: 7DOF-SRS-运动学逆解(几何解析解)实现
date: 2025-01-13
lastmod: 2025-01-13
draft: false
tags: ["Robotics", "Kinematics", "C++"]
categories: ["编程技术"]
authors: ["chase"]
summary: "https://github.com/chase6305/7DofSRSKinematics"
showToc: true
TocOpen: true
hidemeta: false
comments: false
---

你可以到 [https://github.com/chase6305/7DofSRSKinematics](https://github.com/chase6305/7DofSRSKinematics) 查看更详细的介绍.

```python
import numpy as np
from copy import deepcopy
from IPython import embed

class SRSKinSolver:
    def __init__(self):
        self.link_lengths = np.array([0.34, 0.4, 0.4, 0.126])
        half_pi = np.pi / 2
        self.dh_params = np.array(
            [
                [self.link_lengths[0], -half_pi, 0, 0],  # Joint 1
                [0,                     half_pi, 0, 0],  # Joint 2
                [self.link_lengths[1],  half_pi, 0, 0],  # Joint 3
                [0,                    -half_pi, 0, 0],  # Joint 4
                [self.link_lengths[2], -half_pi, 0, 0],  # Joint 5
                [0,                     half_pi, 0, 0],  # Joint 6
                [self.link_lengths[3],  0, 0, 0],  # Joint 7
            ]
        )
        self.d_bs = self.link_lengths[0]
        self.d_se = self.link_lengths[1]
        self.d_ew = self.link_lengths[2]
        self.d_wt = self.link_lengths[3]

    @staticmethod
    def skew(vector: np.ndarray) -> np.ndarray:
        """Compute the skew-symmetric matrix of a vector."""
        return np.array(
            [
                [0, -vector[2], vector[1]],
                [vector[2], 0, -vector[0]],
                [-vector[1], vector[0], 0],
            ]
        )

    def dh_calc(self, d: float, alpha: float, a: float, theta: float) -> np.ndarray:
        """Calculate the transformation matrix based on D-H parameters."""
        T = np.array(
            [
                [
                    np.cos(theta),
                    -np.sin(theta) * np.cos(alpha),
                    np.sin(theta) * np.sin(alpha),
                    a * np.cos(theta),
                ],
                [
                    np.sin(theta),
                    np.cos(theta) * np.cos(alpha),
                    -np.cos(theta) * np.sin(alpha),
                    a * np.sin(theta),
                ],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1],
            ]
        )
        return T

    def configuration(self, rconf: int) -> tuple:
        """Determine the configuration of the arm, elbow, and wrist based on rconf."""
        arm_config = -1 if rconf & 1 else 1
        elbow_config = -1 if rconf & 2 else 1
        wrist_config = -1 if rconf & 4 else 1
        return arm_config, elbow_config, wrist_config

    def calculate_joint_angles(
        self, P_s_to_w: np.ndarray, elbow_GC4: int
    ) -> np.ndarray:
        """Calculate joint angles based on the position from shoulder to wrist and elbow configuration."""
        d_bs, d_se, d_ew = (
            self.link_lengths[0],
            self.link_lengths[1],
            self.link_lengths[2],
        )
        joints = np.zeros(7)

        # Check reachability and calculate elbow joint angle
        norm_P26 = np.linalg.norm(P_s_to_w)
        assert (
            abs(d_bs + d_ew) > norm_P26 > abs(d_bs - d_ew)
        ), "Specified pose outside reachable workspace."

        elbow_cos_angle = (norm_P26**2 - d_se**2 - d_ew**2) / (2 * d_se * d_ew)
        assert abs(elbow_cos_angle) <= 1, "Elbow singularity. End effector at limit."
        joints[3] = elbow_GC4 * np.arccos(elbow_cos_angle)

        # Calculate joint 1
        if np.linalg.norm(P_s_to_w[2]) > 1e-6:
            joints[0] = np.arctan2(P_s_to_w[1], P_s_to_w[0])
        else:
            joints[0] = 0

        # Calculate joint 2
        euclidean_norm = np.hypot(P_s_to_w[0], P_s_to_w[1])
        angle_phi = np.arccos(
            (d_se**2 + norm_P26**2 - d_ew**2) / (2 * d_se * norm_P26)
        )
        joints[1] = (
            np.arctan2(euclidean_norm, P_s_to_w[2]) + elbow_GC4 * angle_phi
        )

        return joints

    def reference_plane(self, pose: np.ndarray, elbow_GC4: int) -> tuple:
        """Calculate the reference plane vector, rotation matrix from base to elbow, and joint values."""
        P_target = pose[:3, 3]
        P02 = np.array([0, 0, self.link_lengths[0]])  # Base to shoulder
        P67 = np.array([0, 0, self.dh_params[-1, 0]])  # Hand to end-effector
        P06 = P_target - pose[:3, :3] @ P67
        P26 = P06 - P02

        # Calculate joint angles
        joint_v = np.zeros(7)
        joint_v = self.calculate_joint_angles(P26, elbow_GC4)

        # Lower arm transformation
        T34_v = np.eye(4)
        T34_v = self.dh_calc(self.dh_params[3, 0], self.dh_params[3, 1], 
                             self.dh_params[3, 2], joint_v[3])
        P34_v = T34_v[:3, 3]
        R34_v = T34_v[:3, :3]

        # Calculate reference elbow position and normal vector to the reference plane
        v1 = (P34_v - P02) / np.linalg.norm(P34_v - P02)
        v2 = (P06 - P02) / np.linalg.norm(P06 - P02)
        V_v_to_sew = np.cross(v1, v2)  # The normal vector to the plane

        R03_v = np.eye(3)
        for i in range(3):
            R03_v = R03_v @ self.dh_calc(
                self.dh_params[i, 0],
                self.dh_params[i, 1],
                self.dh_params[i, 2],
                joint_v[i],
            )[:3,:3]

        return V_v_to_sew, R03_v, joint_v

    def inverse_kinematics(self, pose: np.ndarray, nsparam: float, rconf: int) -> tuple:
        """Perform inverse kinematics to calculate joint angles given a target pose, normalization parameter, and configuration."""
        arm_config, elbow_config, wrist_config = self.configuration(rconf)
        P_target = pose[:3, 3]
        P02 = np.array([0, 0, self.link_lengths[0]])  # Base to shoulder
        P67 = np.array([0, 0, self.dh_params[-1, 0]])  # Hand to end-effector
        P06 = P_target - pose[:3, :3] @ P67
        P26 = P06 - P02

        joints = np.zeros(7)
        # Calculate joint angles
        joints = self.calculate_joint_angles(P26, elbow_config)

        # Calculate transformations
        T34 = self.dh_calc(
            self.dh_params[3, 0], self.dh_params[3, 1], self.dh_params[3, 2], joints[3]
        )
        R34 = T34[:3, :3]

        # Calculate reference plane
        V_v_to_sew, R03_o, joint_v = self.reference_plane(pose, elbow_config)

        # Another way to compute R03_o
        
        # Calculate shoulder joint rotation matrices
        usw = P26 / np.linalg.norm(P26)
        skew_usw = self.skew(usw)

        # angle_psi = np.arctan2(pose[1, 0], pose[0, 0])
        angle_psi = nsparam

        # Calculate rotation matrix R03
        A_s = skew_usw @ R03_o
        B_s = -skew_usw @ skew_usw @ R03_o
        # C_s = (usw @ usw.T) @ R03_o
        C_s = (usw.reshape(-1, 1) @ usw.reshape(1, -1)) @ R03_o  

        # C_s = P26 @ P26 @ R03_o
        R03 = A_s * np.sin(angle_psi) + B_s * np.cos(angle_psi) + C_s

        # Calculate shoulder joint angles
        joints[0] = np.arctan2(R03[1, 1] * arm_config, R03[0, 1] * arm_config)
        joints[1] = np.arccos(R03[2, 1]) * arm_config
        joints[2] = np.arctan2(-R03[2, 2] * arm_config, -R03[2, 0] * arm_config)

        # Calculate wrist joint angles
        A_w = R34.T @ A_s.T @ pose[:3, :3]
        B_w = R34.T @ B_s.T @ pose[:3, :3]
        C_w = R34.T @ C_s.T @ pose[:3, :3]

        # Calculate wrist rotation matrix R47
        R47 = A_w * np.sin(angle_psi) + B_w * np.cos(angle_psi) + C_w

        # Calculate wrist joint angles
        joints[4] = np.arctan2(R47[1, 2] * wrist_config, R47[0, 2] * wrist_config)
        joints[5] = np.arccos(R47[2, 2]) * wrist_config
        joints[6] = np.arctan2(R47[2, 1] * wrist_config, -R47[2, 0] * wrist_config)

        s_mat = np.zeros((3, 3, 3))
        w_mat = np.zeros((3, 3, 3))
        s_mat[:, :, 0] = A_s
        s_mat[:, :, 1] = B_s
        s_mat[:, :, 2] = C_s
        w_mat[:, :, 0] = A_w
        w_mat[:, :, 1] = B_w
        w_mat[:, :, 2] = C_w

        return (
            joints,
            s_mat,
            w_mat,
        )  # Returning joints with placeholders for s_mat and w_mat
        

    def compute_total_transform(self, joint_angles):
        """Compute the overall transformation matrix and the list of transformation matrices for each joint."""
        T_total = np.eye(4)
        T_total_list = []
        for i, params in enumerate(self.dh_params):
            d, alpha, a, theta = params
            if i < len(joint_angles):
                theta += joint_angles[i]

            T = self.dh_calc(d, alpha, a, theta)
            T_total = T_total @ T
            T_total_list.append(T_total.copy())

        return T_total, T_total_list

# Test example
if __name__ == "__main__":
    np.set_printoptions(6, suppress=True)

    ori_joints = np.array([0.0, np.pi/2, 1, np.pi / 2, 1, np.pi / 2, 0])
    kin_solver = SRSKinSolver()

    T_total, T_total_list = kin_solver.compute_total_transform(ori_joints)

    pose = np.array(deepcopy(T_total))
    nsparam = np.pi / 4
    rconf = 0b00000001

    joints, s_mat, w_mat = kin_solver.inverse_kinematics(pose, nsparam, rconf)
    T_total_1, T_total_list_1 = kin_solver.compute_total_transform(joints)

    from IPython import embed
    embed()

```


7DOF-SRS-运动学逆解（几何解析解）实现