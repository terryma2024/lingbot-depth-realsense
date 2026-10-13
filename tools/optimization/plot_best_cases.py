"""Compare the reproduced TensorRT outputs against original level 3."""

import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
VISUALS = ROOT / 'result/optimization/visuals'


def main():
    report = json.loads((ROOT / 'result/optimization/reference-trt-benchmark.json').read_text())
    assert report['status'] == 'PASS'
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 12})
    titles = ['RGB 原图', '原始深度', 'Level 3 BF16', 'TensorRT FP16 / CUDA Graph', '绝对差异（有效像素）']
    figure, axes = plt.subplots(8, 5, figsize=(25, 28), layout='constrained')
    depth_colormap = plt.get_cmap('turbo').copy()
    depth_colormap.set_bad('#151515')
    error_colormap = plt.get_cmap('magma').copy()
    error_colormap.set_bad('#151515')
    manifest = []

    for scene in range(8):
        example_dir = ROOT / 'examples' / str(scene)
        rgb_path = next(example_dir.glob('rgb.*'))
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        raw_depth = cv2.imread(str(example_dir / 'raw_depth.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        baseline = np.load(VISUALS / f'scene_{scene}_depth_level3.npy', allow_pickle=False)
        optimized = np.load(VISUALS / f'scene_{scene}_depth_trt.npy', allow_pickle=False)
        valid = np.isfinite(baseline) & np.isfinite(optimized) & (baseline > 0) & (optimized > 0)
        assert baseline.shape == optimized.shape == (480, 640) and valid.any()
        reference_depths = np.concatenate([
            depth[np.isfinite(depth) & (depth > 0)] for depth in [raw_depth, baseline]
        ])
        minimum, maximum = np.percentile(reference_depths, [1, 99])
        difference = np.abs(optimized - baseline)
        difference_limit = max(float(np.percentile(difference[valid], 99)), 0.001)
        axes[scene, 0].imshow(rgb)
        axes[scene, 0].set_ylabel(f'场景 {scene}', fontsize=16)
        for column, depth in enumerate([raw_depth, baseline, optimized], start=1):
            masked_depth = np.ma.masked_where(~np.isfinite(depth) | (depth <= 0), depth)
            depth_plot = axes[scene, column].imshow(
                masked_depth, cmap=depth_colormap, vmin=minimum, vmax=maximum
            )
        error_plot = axes[scene, 4].imshow(
            np.ma.masked_where(~valid, difference), cmap=error_colormap,
            vmin=0, vmax=difference_limit
        )
        figure.colorbar(depth_plot, ax=axes[scene, 1:4], fraction=0.015, pad=0.01, label='米')
        figure.colorbar(error_plot, ax=axes[scene, 4], fraction=0.035, pad=0.01, label='米')
        for column, axis in enumerate(axes[scene]):
            axis.set_xticks([])
            axis.set_yticks([])
            if scene == 0:
                axis.set_title(titles[column], fontsize=16)
        manifest.append({
            'scene': scene,
            'depth_color_range_m': [float(minimum), float(maximum)],
            'absolute_difference_display_p99_m': difference_limit,
            'baseline': f'scene_{scene}_depth_level3.npy',
            'optimized': f'scene_{scene}_depth_trt.npy',
        })

    figure.suptitle(
        'LingBot-Depth v0.5 · RTX 5060 Ti\n'
        f'Level 3: {report["baseline_mean_ms"]:.2f}ms → TRT + CUDA Graph + 编译投影: '
        f'{report["combined_mean_ms"]:.2f}ms\n'
        '640×480 · 2000 tokens · 同行深度统一色阶；显示范围为 P1–P99\n'
        'TRT 关闭深度 token 掩码、RGB 抗锯齿和输出有效性掩码；普通 TRT 与 CUDA Graph 深度完全一致',
        fontsize=17
    )
    figure.savefig(VISUALS / 'best_optimization_comparison.png', dpi=160)
    figure.savefig(VISUALS / 'best_optimization_comparison_preview.jpg', dpi=80, pil_kwargs={'quality': 92})
    plt.close(figure)
    (VISUALS / 'best_optimization_manifest.json').write_text(
        json.dumps({'scenes': manifest, 'columns': titles, 'accuracy_note': 'No ground-truth depth is available.'},
                   ensure_ascii=False, indent=2) + '\n'
    )
    print('Saved', VISUALS / 'best_optimization_comparison.png')


if __name__ == '__main__':
    main()
