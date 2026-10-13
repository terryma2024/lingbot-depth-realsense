"""Experimental benchmark only; does not modify model source or saved weights."""
import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import xformers.ops.fmha as fmha
fmha._set_use_fa3(False)
from mdm.model.v2 import MDMModel
from example import preprocess_input_image, load_depth_map, load_intrinsics


def inputs(scene):
    directory = ROOT / 'examples' / str(scene)
    rgb = next(directory.glob('rgb.*'))
    _, image = preprocess_input_image(str(rgb), torch.device('cuda'))
    depth = torch.as_tensor(load_depth_map(str(directory / 'raw_depth.png')), device='cuda')
    intrinsics = torch.as_tensor(load_intrinsics(str(directory / 'intrinsics.txt'), 640, 480), device='cuda')[None]
    return image, depth, intrinsics


def infer(model, data):
    torch.compiler.cudagraph_mark_step_begin()
    image, depth, intrinsics = data
    return model.infer(image, depth_in=depth, intrinsics=intrinsics, resolution_level=3)


def compare(output, reference):
    a = output['depth'].detach().cpu()
    b = reference['depth']
    assert a.shape == b.shape == (1, 480, 640)
    assert not torch.isnan(a).any()
    valid = torch.isfinite(a) & torch.isfinite(b) & (a > 0) & (b > 0)
    assert valid.any()
    assert (a[torch.isfinite(a)] > 0).all()
    delta = (a[valid] - b[valid]).abs()
    assert output['points'].shape == (1, 480, 640, 3)
    torch.testing.assert_close(output['points'][..., 2], output['depth'], rtol=0, atol=0)
    return {'mean_abs_difference_m': delta.mean().item(),
            'p99_abs_difference_m': torch.quantile(delta, .99).item(),
            'mean_relative_difference_pct': (delta / b[valid]).mean().item() * 100,
            'mask_disagreement_pct': (output['mask'].cpu() != reference['mask']).float().mean().item() * 100,
            'valid_output_pct': torch.isfinite(a).float().mean().item() * 100}


def measure(model, data, repeats=20):
    for _ in range(3):
        infer(model, data)
    torch.cuda.synchronize()
    timings = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        infer(model, data)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    return {'mean_ms': statistics.mean(timings), 'median_ms': statistics.median(timings),
            'p95_ms': sorted(timings)[int(.95 * (len(timings)-1))], 'runs': repeats}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scope', choices=['decoder', 'forward'], default='decoder')
    p.add_argument('--mode', default='reduce-overhead')
    args = p.parse_args()
    model = MDMModel.from_pretrained(ROOT / 'checkpoints/lingbot-depth-v0.5/model.pt').cuda().eval()
    data = [inputs(i) for i in range(8)]
    references = []
    baseline = []
    for i, frame in enumerate(data):
        result = infer(model, frame)
        references.append({k: v.detach().cpu().clone() for k, v in result.items()})
        del result
        baseline.append({'scene': i, **measure(model, frame)})
    print('BASELINE', json.dumps(baseline), flush=True)
    started = time.perf_counter()
    if args.scope == 'decoder':
        for name in ['neck', 'depth_head', 'mask_head']:
            getattr(model, name).compile(mode=args.mode, dynamic=False)
    else:
        model.forward = torch.compile(model.forward, mode=args.mode, dynamic=True)
    print('COMPILING', args.scope, args.mode, flush=True)
    out = infer(model, data[0])
    torch.cuda.synchronize()
    cold = time.perf_counter() - started
    print('COLD_COMPILE_SECONDS', cold, flush=True)
    del out
    results = []
    for i, frame in enumerate(data):
        out = infer(model, frame)
        differences = compare(out, references[i])
        del out
        timings = measure(model, frame)
        row = {'scene': i, **timings, **differences}
        results.append(row)
        print('RESULT', json.dumps(row), flush=True)
    report = {'scope': args.scope, 'mode': args.mode, 'level': 3,
              'input_and_output_resolution': [640, 480], 'batch_size': 1,
              'precision': 'bfloat16 autocast', 'cold_compile_seconds_excluded': cold,
              'measurement': 'GPU-resident inputs; synchronized host wall time; preprocessing, loading and compilation excluded',
              'baseline': baseline, 'compiled': results,
              'baseline_mean_ms': statistics.mean(r['mean_ms'] for r in baseline),
              'compiled_mean_ms': statistics.mean(r['mean_ms'] for r in results),
              'limitations': 'Comparison is against eager level 3, not ground truth.'}
    destination = ROOT / f'result/optimization/compile-{args.scope}-{args.mode}.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print('REPORT', destination, flush=True)


if __name__ == '__main__':
    main()
