"""Create one publication-style comparison grid from measured model outputs."""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
VIS = ROOT / 'result/optimization/visuals'


def main():
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 15,
                         'axes.unicode_minus': False, 'savefig.facecolor': 'white'})
    columns = [
        ('rgb', 'RGB 原图'),
        ('raw', '原始 Depth'),
        ('level9', '默认 Level 9\nBF16'),
        ('level3', 'Level 3 基线\nBF16'),
        ('compile_decoder_reduce-overhead', 'L3 编译解码器\nreduce-overhead'),
        ('compile_forward_default', 'L3 编译前向\ndefault'),
        ('compile_forward_reduce-overhead', 'L3 编译前向\nreduce-overhead'),
        ('compile_forward_max-autotune', 'L3 编译前向\nmax-autotune'),
        ('trt', 'L3 TensorRT\nFP16'),
        ('trt_combined', 'L3 TRT + CUDA Graph\n编译后处理'),
    ]
    cmap = plt.get_cmap('turbo').copy()
    cmap.set_bad('#151515')
    fig, axes = plt.subplots(8, len(columns), figsize=(41, 26))
    fig.subplots_adjust(left=.034, right=.962, bottom=.078, top=.923, wspace=.022, hspace=.065)
    ranges, sources = [], []
    for scene in range(8):
        rgb_path = next((ROOT / f'examples/{scene}').glob('rgb.*'))
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        raw = cv2.imread(str(ROOT / f'examples/{scene}/raw_depth.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        level9 = np.load(ROOT / f'result/example_{scene}/depth_refined.npy')
        level3 = np.load(VIS / f'scene_{scene}_depth_level3.npy')
        trt = np.load(VIS / f'scene_{scene}_depth_trt.npy')
        data = {'rgb': rgb, 'raw': raw, 'level9': level9, 'level3': level3, 'trt': trt,
                'trt_combined': trt}
        for name, _ in columns:
            if name.startswith('compile_'):
                data[name] = np.load(VIS / f'scene_{scene}_depth_{name}.npy')
        # Let the original data and baselines set the range. TRT-only unmasked
        # far-background values must not flatten the contrast of every panel.
        reference_values = np.concatenate([a[np.isfinite(a) & (a > 0)] for a in [raw, level9, level3]])
        lo, hi = np.percentile(reference_values, [1, 99])
        ranges.append({'scene': scene, 'depth_color_min_m': float(lo), 'depth_color_max_m': float(hi)})
        for col, (name, title) in enumerate(columns):
            ax = axes[scene, col]
            if name == 'rgb':
                ax.imshow(rgb)
                ax.set_ylabel(f'Case {scene}', rotation=0, ha='right', va='center', labelpad=14,
                              fontsize=21, fontweight='bold')
            else:
                depth = data[name]
                assert depth.shape == (480, 640) and not np.isnan(depth).any()
                masked = np.ma.masked_where(~np.isfinite(depth) | (depth <= 0), depth)
                plot = ax.imshow(masked, cmap=cmap, vmin=lo, vmax=hi, interpolation='nearest')
            if scene == 0:
                ax.set_title(title, fontsize=18, pad=15, fontweight='medium')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        # One shared colorbar for all nine depth panels in this row.
        bbox = axes[scene, -1].get_position()
        bar = fig.add_axes([.967, bbox.y0 + .009, .004, bbox.height - .018])
        colorbar = fig.colorbar(plot, cax=bar)
        colorbar.ax.tick_params(labelsize=14, length=3)
        colorbar.set_ticks([lo, (lo + hi) / 2, hi])
        colorbar.set_ticklabels([f'{lo:.2f}', f'{(lo+hi)/2:.2f}', f'{hi:.2f}'])
        sources.append({'scene': scene, 'rgb': str(rgb_path.relative_to(ROOT)),
                        'raw_depth': f'examples/{scene}/raw_depth.png',
                        'level9': f'result/example_{scene}/depth_refined.npy',
                        'level3': f'result/optimization/visuals/scene_{scene}_depth_level3.npy',
                        'trt': f'result/optimization/visuals/scene_{scene}_depth_trt.npy',
                        'compiled_outputs': [f'result/optimization/visuals/scene_{scene}_depth_{name}.npy'
                                             for name, _ in columns if name.startswith('compile_')]})
    fig.suptitle('LingBot-Depth v0.5 ｜ 全部 8 个场景：原图、基线与优化方案', fontsize=31, y=.984, fontweight='bold')
    fig.text(.5, .952, '每个 Case 一行；同一行所有深度图共用右侧色阶（单位：米）；每格输入/输出尺寸为 640 × 480',
             ha='center', fontsize=20)
    fig.text(.04, .047, '色阶：每行原始 Depth + Level 9 + Level 3 有效深度的 P1–P99，超出范围的值只在显示时截断；黑色表示无效深度。', fontsize=18)
    fig.text(.04, .027, 'TRT 参考方案采用 FP16，并关闭深度 token 掩码、RGB 抗锯齿插值和输出有效性掩码；最后两列深度输出已验证逐像素一致。', fontsize=18)
    fig.text(.04, .010, 'Level 3 的目标 token 数为 2000；编译结果是重新运行对应已测试配置所得，数值差异已与先前报告核对。', fontsize=18)
    image_path = VIS / 'all_cases_all_optimizations.png'
    fig.savefig(image_path, dpi=180)
    # A lighter preview preserves the same complete layout.
    fig.savefig(VIS / 'all_cases_all_optimizations_preview.jpg', dpi=90, pil_kwargs={'quality': 92})
    plt.close(fig)
    manifest = {'png': image_path.name, 'preview': 'all_cases_all_optimizations_preview.jpg',
                'rows': 8, 'columns': [{'key': k, 'title': t} for k, t in columns],
                'color_scales': ranges, 'sources': sources,
                'trt_combined': 'Same saved depth used in both TRT columns because benchmark verified exact equality between ordinary and CUDA Graph execution.',
                'precision_and_mask_note': 'TRT differs from original inference as annotated on the figure. No ground-truth depth is present.'}
    (VIS / 'all_cases_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print('Saved', image_path)


if __name__ == '__main__':
    main()
