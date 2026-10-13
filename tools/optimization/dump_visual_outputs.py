"""Regenerate predictions from previously benchmarked compile configurations."""
import argparse
import json
import numpy as np
import torch
from probe_compile import ROOT, MDMModel, inputs, infer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=['decoder', 'forward'], required=True)
    parser.add_argument('--mode', required=True)
    args = parser.parse_args()
    model = MDMModel.from_pretrained(ROOT / 'checkpoints/lingbot-depth-v0.5/model.pt').cuda().eval()
    if args.scope == 'decoder':
        for name in ['neck', 'depth_head', 'mask_head']:
            getattr(model, name).compile(mode=args.mode, dynamic=False)
    else:
        model.forward = torch.compile(model.forward, mode=args.mode, dynamic=True)
    prior = json.loads((ROOT / f'result/optimization/compile-{args.scope}-{args.mode}.json').read_text())
    for scene in range(8):
        result = infer(model, inputs(scene))
        depth = result['depth'][0].float().cpu().numpy()
        del result
        baseline = np.load(ROOT / f'result/optimization/visuals/scene_{scene}_depth_level3.npy')
        valid = np.isfinite(depth) & np.isfinite(baseline) & (depth > 0) & (baseline > 0)
        assert depth.shape == (480, 640) and not np.isnan(depth).any() and valid.any()
        mean_relative = float((np.abs(depth[valid] - baseline[valid]) / baseline[valid]).mean() * 100)
        expected = prior['compiled'][scene]['mean_relative_difference_pct']
        assert abs(mean_relative - expected) < .02, (scene, mean_relative, expected)
        path = ROOT / f'result/optimization/visuals/scene_{scene}_depth_compile_{args.scope}_{args.mode}.npy'
        np.save(path, depth)
        print(path.name, 'verified; mean relative difference', mean_relative, flush=True)


if __name__ == '__main__':
    main()
