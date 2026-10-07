"""Cross-check pytorch-kinematics 0.10.0 on the accompanying planar2r.urdf.

Python 3.10+, PyTorch, pytorch-kinematics. CPU float64, no robot/GPU access.
Uses closed-form geometry, central differences, gradcheck and gradgradcheck.
"""
import argparse
import json
from importlib.metadata import version
from pathlib import Path

import torch
import pytorch_kinematics as pk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf', type=Path, default=Path(__file__).with_name('planar2r.urdf'))
    parser.add_argument('--output', type=Path, default=Path('gradient-results.json'))
    args = parser.parse_args()
    dtype = torch.float64
    chain = pk.build_serial_chain_from_urdf(args.urdf.read_text(), 'tip').to(dtype=dtype, device='cpu')
    assert chain.get_joint_parameter_names() == ['shoulder', 'elbow']
    tip = int(chain.get_frame_indices('tip').item())

    def position(q, analytical_grad=True):
        all_frames = chain.forward_kinematics_tensor(q.unsqueeze(0), analytical_grad=analytical_grad)
        return all_frames[tip, 0, :3, 3]

    def analytic(q):
        q1, q2 = q
        c1, s1 = torch.cos(q1), torch.sin(q1)
        c12, s12 = torch.cos(q1+q2), torch.sin(q1+q2)
        point = torch.stack((.8*c1+.6*c12, .8*s1+.6*s12, q1*0))
        jacobian = torch.stack((torch.stack((-.8*s1-.6*s12, -.6*s12)),
                                torch.stack((.8*c1+.6*c12, .6*c12)), q*0))
        hx = torch.stack((torch.stack((-.8*c1-.6*c12, -.6*c12)),
                          torch.stack((-.6*c12, -.6*c12))))
        hy = torch.stack((torch.stack((-.8*s1-.6*s12, -.6*s12)),
                          torch.stack((-.6*s12, -.6*s12))))
        return point, jacobian, hx, hy

    rows = []
    for values in ([.4, -.7], [-.8, 1.1], [0., 0.], [1.2, -2.]):
        q = torch.tensor(values, dtype=dtype, requires_grad=True)
        exact_p, exact_j, _, _ = analytic(q)
        jacobian = chain.jacobian(q.unsqueeze(0))[0]
        automatic = torch.autograd.functional.jacobian(position, q)
        ordinary = torch.autograd.functional.jacobian(lambda x: position(x, False), q)
        perturbations = torch.eye(2, dtype=dtype)*1e-6
        numeric = torch.stack([(position(q+step)-position(q-step))/2e-6
                               for step in perturbations], dim=1)
        torch.testing.assert_close(jacobian[:3], automatic, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(automatic, ordinary, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(automatic, numeric, atol=1e-9, rtol=1e-8)
        # URDF transforms are initially parsed in float32 in this package version.
        # Converting the completed chain to float64 does not undo that rounding.
        torch.testing.assert_close(position(q), exact_p, atol=5e-8, rtol=0)
        torch.testing.assert_close(jacobian[:3], exact_j, atol=5e-8, rtol=0)
        angular_expected = torch.tensor([[0., 0.], [0., 0.], [1., 1.]], dtype=dtype)
        torch.testing.assert_close(jacobian[3:], angular_expected, atol=1e-12, rtol=0)
        rows.append({'q_rad': values,
                     'jacobian_finite_difference_max_error': float((automatic-numeric).abs().max().detach()),
                     'position_decimal_geometry_max_error_m': float((position(q)-exact_p).abs().max().detach()),
                     'jacobian_closed_form_max_error': float((jacobian[:3]-exact_j).abs().max().detach())})

    q = torch.tensor([.4, -.7], dtype=dtype, requires_grad=True)
    desired = torch.tensor([.9, .3, 0.], dtype=dtype)
    loss = lambda x: .5*((position(x, False)-desired)**2).sum()
    hessian = torch.autograd.functional.hessian(loss, q)
    point, jacobian, hx, hy = analytic(q)
    error = point-desired
    expected_hessian = jacobian.T @ jacobian + error[0]*hx + error[1]*hy
    torch.testing.assert_close(hessian, expected_hessian, atol=1e-7, rtol=0)
    first_order = torch.autograd.gradcheck(position, (q,), eps=1e-6, atol=1e-6, rtol=1e-5)
    second_order = torch.autograd.gradgradcheck(lambda x: position(x, False), (q,),
                                               eps=1e-6, atol=1e-6, rtol=1e-5)
    report = {'torch': torch.__version__, 'pytorch_kinematics': version('pytorch-kinematics'),
              'device': 'cpu', 'dtype': 'float64', 'cases': rows,
              'gradcheck_default': first_order, 'gradgradcheck_standard_autograd': second_order,
              'loss_hessian': hessian.detach().tolist(),
              'closed_form_hessian_max_error': float((hessian-expected_hessian).abs().max().detach()),
              'scope': 'Two bounded revolute joints and one fixed tool; position FK derivatives and yaw geometric Jacobian. Does not differentiate through IK or calibrate URDF parameters.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
