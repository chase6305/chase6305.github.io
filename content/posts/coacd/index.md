---
title: "CoACD: 基于碰撞感知凹性与树搜索的近似凸分解"
date: 2025-04-03
lastmod: 2025-04-03
draft: false
tags: ["CoACD"]
categories: ["CoACD"]
authors: ["chase"]
summary: "CoACD: 基于碰撞感知凹性与树搜索的近似凸分解"
showToc: true
TocOpen: true
hidemeta: false
comments: false
---

### CoACD 基于碰撞感知凹性与树搜索的近似凸分解
- [CoACD 官方文档](https://colin97.github.io/CoACD/)

CoACD（Convex Approximation of Complex Decompositions）是一种用于将复杂网格分解为多个凸包的算法, 专为 3D 网格设计了近似凸分解算法，强调在保持物体间潜在碰撞条件的同时减少组件数量，优化了后续应用中的精细且高效的对象交互。该算法在计算机图形学、物理模拟和碰撞检测等领域有广泛应用。

本文档将详细介绍如何使用 Open3D 可视化 CoACD 凸包分解前后的网格。

#### 1. 安装依赖
首先，确保你已经安装了必要的依赖库：
```bash
pip install trimesh open3d coacd
```

#### 2. 加载和处理网格
我们将使用 `trimesh` 加载一个网格文件，并使用 `coacd` 进行凸包分解。

```python
import coacd
import trimesh
import open3d as o3d
import numpy as np

# 加载输入的网格文件
input_file = 'doll.obj'
mesh = trimesh.load(input_file, force="mesh")

# 将加载的网格转换为 coacd 的 Mesh 对象
mesh_coacd = coacd.Mesh(mesh.vertices, mesh.faces)

# 运行 CoACD 算法，返回凸包列表
parts = coacd.run_coacd(mesh_coacd)
```

#### 3. 使用 Open3D 可视化原始网格
我们将使用 Open3D 可视化原始的网格。

```python
# 使用 open3d 可视化原始网格
original_mesh = o3d.geometry.TriangleMesh(
    vertices=o3d.utility.Vector3dVector(mesh.vertices),
    triangles=o3d.utility.Vector3iVector(mesh.faces)
)
original_mesh.paint_uniform_color([0.5, 0.5, 0.5])
o3d.visualization.draw_geometries([original_mesh], window_name="Original Mesh")
```

#### 4. 使用 Open3D 可视化凸包分解后的网格
接下来，我们将使用 Open3D 可视化凸包分解后的网格。

```python
# 使用 open3d 可视化凸包分解后的网格
convex_meshes = []
for part in parts:
    vertices = part[0]
    faces = part[1]
    convex_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(faces)
    )
    convex_mesh.paint_uniform_color(np.random.random(3))
    convex_meshes.append(convex_mesh)

o3d.visualization.draw_geometries(convex_meshes, window_name="Convex Decomposition")
```
#### 5. 整体示例
```python
import coacd
import trimesh
import open3d as o3d
import numpy as np

# 加载输入的网格文件
input_file = 'doll.obj'
mesh = trimesh.load(input_file, force="mesh")

# 将加载的网格转换为 coacd 的 Mesh 对象
mesh_coacd = coacd.Mesh(mesh.vertices, mesh.faces)

# 运行 CoACD 算法，返回凸包列表
parts = coacd.run_coacd(mesh_coacd)

# 使用 open3d 可视化原始网格
original_mesh = o3d.geometry.TriangleMesh(
    vertices=o3d.utility.Vector3dVector(mesh.vertices),
    triangles=o3d.utility.Vector3iVector(mesh.faces)
)
original_mesh.paint_uniform_color([0.5, 0.5, 0.5])
o3d.visualization.draw_geometries([original_mesh], window_name="Original Mesh")

# 使用 open3d 可视化凸包分解后的网格
convex_meshes = []
for part in parts:
    vertices = part[0]
    faces = part[1]
    convex_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(faces)
    )
    convex_mesh.paint_uniform_color(np.random.random(3))
    convex_meshes.append(convex_mesh)

o3d.visualization.draw_geometries(convex_meshes, window_name="Convex Decomposition")
```
![ori_coacd](ori_coacd.jpeg)

![coacd](coacd.jpeg) 

### 5. CoACD 与 V-HACD 的区别

CoACD（Convex Approximation of Complex Decompositions）和 V-HACD（Volumetric Hierarchical Approximate Convex Decomposition）都是用于将复杂网格分解为多个凸包的算法。尽管它们的目标相似，但在实现细节和应用场景上存在一些区别。

#### 5.1 算法原理对比

- **CoACD**：
  - **初始分解**：通过初步分解生成初始的凸包集合。
  - **迭代优化**：通过迭代优化过程，逐步改进凸包的质量，使其更好地逼近原始网格。
  - **合并与细化**：在优化过程中，可能会对凸包进行合并或细化，以提高分解的质量。

- **V-HACD**：
  - **体积分解**：基于体积的层次化分解方法，将网格分解为多个体积块。
  - **层次化聚类**：通过层次化聚类算法，将体积块聚合成凸包。
  - **细化与优化**：对生成的凸包进行细化和优化，以提高分解的质量。

#### 5.2 适用场景

- **CoACD**：
  - 更适合需要高精度凸包分解的场景。
  - 在需要迭代优化和细化的应用中表现更好。
  - 适用于需要动态调整凸包数量和精度的场景。

- **V-HACD**：
  - 更适合需要快速生成凸包分解的场景。
  - 在处理大规模网格时表现更好。
  - 适用于需要层次化分解和聚类的应用。

#### 5.3 性能对比

- **CoACD**：
  - 由于迭代优化过程，计算时间可能较长。
  - 生成的凸包质量较高，适合高精度需求。

- **V-HACD**：
  - 计算速度较快，适合实时应用。
  - 生成的凸包数量较多，适合大规模网格。

#### 5.4 使用方法

- **CoACD**：
  ```python
  import coacd
  import trimesh

  # 加载网格
  mesh = trimesh.load('doll.obj', force="mesh")
  mesh_coacd = coacd.Mesh(mesh.vertices, mesh.faces)

  # 运行 CoACD
  parts = coacd.run_coacd(mesh_coacd)
  ```

- **V-HACD**：
  ```python
  import pyvhacd
  import trimesh

  # 加载网格
  mesh = trimesh.load('doll.obj', force="mesh")

  # 运行 V-HACD
  vhacd = pyvhacd.VHACD()
  vhacd.compute(mesh.vertices, mesh.faces)
  parts = vhacd.get_convex_hulls()
  ```

#### 5.5 参数设置

- **CoACD**：
  - `max_convex_hull`: 最大凸包数量。
  - `threshold`: 精度阈值。
  - `max_iter`: 最大迭代次数。

- **V-HACD**：
  - `resolution`: 分辨率。
  - `concavity`: 凸度。
  - `plane_downsampling`: 平面降采样。
  - `convex_hull_downsampling`: 凸包降采样。

#### 5.6 总结

- **CoACD** 更适合需要高精度和迭代优化的应用。
- **V-HACD** 更适合需要快速分解和处理大规模网格的应用。

根据具体需求选择合适的算法，可以更好地满足应用场景的要求。