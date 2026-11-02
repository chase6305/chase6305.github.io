"""Compare pinned VLA padding with an explicit private-zero candidate.

Two or three CPU/Gloo processes execute three tiny steps per configuration.
Both variants redirect CUDA allocation to CPU and use a storage-sharing flat
Gloo output view. Completion is either natural or waited before wrapper return.
The candidate modifies two assignments in memory; the official file is unchanged.
This is not a model, multi-GPU or performance reproduction.
"""
import argparse
import ast
from datetime import timedelta
import hashlib
import importlib.util
import json
import math
import multiprocessing
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

SHA = '8e7ca8d6f40ebc4fb066acb15387e618d515896cca5d1eb0ad234e5310e9157e'


def require(condition, message):
    """Keep verification enabled under optimized Python too."""
    if not condition:
        raise RuntimeError(message)


def worker(rank, world, source, folder):
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    spec = importlib.util.spec_from_file_location('official_padding_probe', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.DefaultPion
    tree = ast.parse(Path(source).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DefaultPion')

    class IndependentZero(ast.NodeTransformer):
        def __init__(self):
            self.lines = []

        def visit_Assign(self, value):
            if (len(value.targets) == 1 and isinstance(value.targets[0], ast.Name)
                    and value.targets[0].id == 'g' and isinstance(value.value, ast.Subscript)
                    and isinstance(value.value.value, ast.Name)
                    and value.value.value.id == 'update_buffer_views'):
                self.lines.append(value.lineno)
                value.value = ast.Call(func=ast.Attribute(value=ast.Name(id='torch', ctx=ast.Load()),
                                                        attr='zeros_like', ctx=ast.Load()),
                                       args=[value.value], keywords=[])
            return self.generic_visit(value)

    transform = IndependentZero()
    fixed_node = transform.visit(node)
    require(transform.lines == [286, 299], f'Unexpected patch sites: {transform.lines}')
    namespace = dict(vars(module))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fixed_node], type_ignores=[])),
                 '<private-zero-variant>', 'exec'), namespace)
    fixed = namespace['DefaultPion']
    empty, gather = torch.empty, dist.all_gather_into_tensor

    def cpu_buffer(*args, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs = {**kwargs, 'device': 'cpu'}
        return empty(*args, **kwargs)

    configurations = [
        {'name': 'one_full_batch', 'count': world, 'tail': 'nonzero', 'separate_last': False},
        {'name': 'padding_all_gradients', 'count': world + 1, 'tail': 'nonzero', 'separate_last': False},
        {'name': 'two_batches_then_padding', 'count': 2 * world + 1, 'tail': 'nonzero', 'separate_last': False},
        {'name': 'missing_gradient_and_padding', 'count': world + 1, 'tail': 'none', 'separate_last': False},
        {'name': 'zero_gradient_and_padding', 'count': world + 1, 'tail': 'zero', 'separate_last': False},
        {'name': 'different_numel_group', 'count': world + 1, 'tail': 'nonzero', 'separate_last': True},
    ]
    dist.init_process_group('gloo', init_method=(Path(folder) / 'rendezvous').as_uri(),
                            rank=rank, world_size=world, timeout=timedelta(seconds=20))
    rows = []
    try:
        for completion in ['natural', 'complete_before_return']:
            def flat_gather(output, value, *, async_op):
                result = gather(output.view(-1), value, async_op=async_op)
                if completion == 'complete_before_return':
                    result.wait()
                return result

            for variant, cls in [('original', original), ('private_zero', fixed)]:
                for configuration in configurations:
                    dist.barrier()
                    params, expected = [], []
                    for index in range(configuration['count']):
                        shape = (2, 3) if configuration['separate_last'] and index == configuration['count'] - 1 else (2, 2)
                        parameter = torch.nn.Parameter(torch.full(shape, float(index + 1), dtype=torch.float64))
                        params.append(parameter)
                        expected.append(parameter.detach().clone())
                    with patch.object(torch, 'empty', cpu_buffer):
                        optimizer = cls(params, lr=.01, weight_decay=.1, momentum=.5,
                                        nesterov=False, promotion_steps=1, ns_steps=5,
                                        scale_factor=.2, rank=rank, world_size=world)
                    owners = {id(parameter): position % world
                              for group in optimizer.param_groups
                              for position, parameter in enumerate(group['params'])}
                    # The reference uses the closed-form direction of a signed
                    # coordinate rank-one matrix, independently of helper code.
                    # Step 2 observes zero but retains historical momentum;
                    # step 3 has no observation and applies only weight decay.
                    amplitudes = [0.] * len(params)
                    observed_before = [False] * len(params)
                    steps = []
                    for step in range(1, 4):
                        kinds = []
                        for index, parameter in enumerate(params):
                            kind = (configuration['tail'] if index == len(params) - 1 else 'nonzero') if step == 1 else 'zero' if step == 2 else 'none'
                            kinds.append(kind)
                            i, j = (index // 2) % 2, index % 2
                            sign = -1. if index % 3 == 1 else 1.
                            expected[index] *= .999
                            if kind == 'none':
                                parameter.grad = None
                                continue
                            parameter.grad = torch.zeros_like(parameter)
                            observed_before[index] = True
                            observed = sign if kind == 'nonzero' else 0.
                            parameter.grad[i, j] = observed
                            amplitudes[index] = .5 * amplitudes[index] + .5 * observed
                            if amplitudes[index] != 0.:
                                expected[index][i, j] -= .01 * .2 * math.sqrt(max(parameter.shape)) * math.copysign(1., amplitudes[index])
                        with patch.object(dist, 'all_gather_into_tensor', flat_gather):
                            optimizer.step()
                        momentum_errors, momentum_indices = [], []
                        for index, parameter in enumerate(params):
                            buffer = optimizer.state.get(parameter, {}).get('momentum_buffer')
                            present = owners[id(parameter)] == rank and observed_before[index]
                            require((buffer is not None) == present, 'Unexpected momentum state ownership')
                            if present:
                                reference = torch.zeros_like(parameter)
                                reference[(index // 2) % 2, index % 2] = amplitudes[index]
                                momentum_errors.append(float((buffer - reference).abs().max()))
                                momentum_indices.append(index)
                        require(max(momentum_errors, default=0.) == 0., 'Momentum state disagrees with the independent recurrence')
                        differences = [float((p.detach() - e).abs().max()) for p, e in zip(params, expected)]
                        steps.append({
                            'step': step, 'gradient_kinds': kinds,
                            'max_weight_differences': differences,
                            'matches_independent_updates': max(differences) < 4e-15,
                            'owned_momentum_indices': momentum_indices,
                            'max_momentum_error': max(momentum_errors, default=0.),
                            'actual_weights': [p.detach().tolist() for p in params],
                            'expected_weights': [e.tolist() for e in expected],
                        })
                        if variant == 'private_zero':
                            require(max(differences) < 4e-15, f'Candidate disagrees: {steps[-1]}')
                    rows.append({
                        'rank': rank, 'world_size': world, 'completion': completion,
                        'variant': variant, **configuration,
                        'steps': steps,
                    })
                    dist.barrier()
    finally:
        dist.destroy_process_group()
    (Path(folder) / f'rank-{rank}.json').write_text(json.dumps({'torch': torch.__version__,
        'patched_zero_assignments': transform.lines, 'cases': rows}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-file', type=Path, required=True)
    parser.add_argument('--world-size', type=int, choices=[2, 3], default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        source_hash = hashlib.sha256(args.official_file.read_bytes()).hexdigest()
    except OSError as error:
        parser.error(str(error))
    if source_hash != SHA:
        parser.error('Official VLA source does not match pinned Pion commit 8796501')
    # Reject source aliases before importing the optional dependency or starting
    # any workers. A successful report must never replace the audited source.
    if (args.output.resolve() == args.official_file.resolve()
            or args.output.exists() and args.output.samefile(args.official_file)):
        parser.error('--output must not refer to the official source file')
    if not args.output.parent.is_dir() or args.output.is_dir():
        parser.error('--output needs an existing parent directory and a file path')
    if importlib.util.find_spec('torch') is None:
        parser.error('This optional probe requires PyTorch with the Gloo CPU backend')
    ctx = multiprocessing.get_context('spawn')
    with tempfile.TemporaryDirectory(prefix='pion-padding-') as folder:
        processes = [ctx.Process(target=worker, args=(rank, args.world_size, str(args.official_file), folder))
                     for rank in range(args.world_size)]
        for process in processes:
            process.start()
        deadline = time.monotonic() + 60
        try:
            for process in processes:
                process.join(max(0, deadline - time.monotonic()))
            require(all(process.exitcode == 0 for process in processes),
                    f'Gloo worker failed or timed out: {[p.exitcode for p in processes]}')
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(2)
        ranks = [json.loads((Path(folder) / f'rank-{rank}.json').read_text()) for rank in range(args.world_size)]
    summary = []
    for index, first in enumerate(ranks[0]['cases']):
        rows = [rank['cases'][index] for rank in ranks]
        step_results = []
        for step_index in range(3):
            states = [row['steps'][step_index] for row in rows]
            disagreement = max(abs(a - b) for state in states[1:]
                               for m1, m2 in zip(states[0]['actual_weights'], state['actual_weights'])
                               for r1, r2 in zip(m1, m2) for a, b in zip(r1, r2))
            step_results.append({'step': step_index + 1,
                'ranks_with_wrong_updates': [row['rank'] for row in rows if not row['steps'][step_index]['matches_independent_updates']],
                'rank_weight_disagreement': disagreement,
                'maximum_weight_error': max(max(state['max_weight_differences']) for state in states)})
        if first['name'] in ['one_full_batch', 'different_numel_group']:
            require(all(not step['ranks_with_wrong_updates'] for step in step_results), 'Independent controls failed')
        if first['variant'] == 'original' and first['name'] == 'padding_all_gradients' and first['completion'] == 'complete_before_return':
            require(step_results[0]['ranks_with_wrong_updates'] == list(range(1, args.world_size)), 'Padding counterexample was not reproduced')
            require(step_results[0]['rank_weight_disagreement'] > .002, 'Expected rank disagreement absent')
        summary.append({k: first[k] for k in ['world_size', 'completion', 'variant', 'name', 'count', 'tail', 'separate_last']}
                       | {'steps': step_results})
    require(hashlib.sha256(args.official_file.read_bytes()).hexdigest() == SHA, 'Official source file changed')
    report = {
        'source_commit': '87965014b360bde604632cba0aa12e423c6c63c6',
        'source_sha256': SHA, 'source_file_modified': False,
        'world_size': args.world_size, 'backend': 'gloo', 'device': 'cpu',
        'parameter_dtype': 'float64', 'internal_dtype': 'bfloat16',
        'variants': {'original': {'in_memory_class_modified': False},
                     'private_zero': {'in_memory_class_modified': True, 'patch_lines': [286, 299],
                                      'change': 'Wrap the receive-buffer input view in torch.zeros_like at both missing-gradient and padding sites'}},
        'completion_schedules': {'natural': 'Original asynchronous Gloo return',
                                 'complete_before_return': 'Wrapper waits for Gloo completion before returning its handle; forces a valid completed-before-next-iteration ordering'},
        'settings': {'lr': .01, 'weight_decay': .1, 'momentum': .5, 'nesterov': False,
                     'promotion_steps': 1, 'ns_steps': 5, 'scale_factor': .2},
        'steps': ['configured gradient observations', 'zero gradients with historical momentum', 'missing gradients, decay only'],
        'momentum_checks': 'Each step independently checks state ownership, initialization and EMA values; a missing observation must preserve existing state.',
        'ranks': ranks, 'summary': summary,
        'scope': 'Original and candidate classes use CPU allocation plus a storage-sharing flat Gloo output view. Completion is natural or explicitly completed before wrapper return. Tiny signed rank-one coordinate observations; no model, CUDA/NCCL multi-rank execution or performance benchmark.',
    }
    encoded = json.dumps(report, indent=2, allow_nan=False) + '\n'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                dir=args.output.parent, prefix='.pion-padding-', suffix='.tmp',
                delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
        temporary.replace(args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({'world_size': args.world_size, 'cases': len(summary),
                      'optimizer_step_calls': len(summary) * args.world_size * 3,
                      'candidate_cases_with_wrong_updates': sum(row['variant'] == 'private_zero' and any(step['ranks_with_wrong_updates'] for step in row['steps']) for row in summary),
                      'original_cases_with_wrong_updates': sum(row['variant'] == 'original' and any(step['ranks_with_wrong_updates'] for step in row['steps']) for row in summary)}))


if __name__ == '__main__':
    main()
