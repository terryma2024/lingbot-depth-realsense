"""Save original level-3 predictions and already measured TensorRT predictions."""
import numpy as np
import torch
from probe_compile import ROOT, MDMModel, inputs, infer


def main():
    target = ROOT / 'result/optimization/visuals'
    target.mkdir(parents=True, exist_ok=True)
    model = MDMModel.from_pretrained(ROOT / 'checkpoints/lingbot-depth-v0.5/model.pt').cuda().eval()
    for scene in range(8):
        output = infer(model, inputs(scene))
        baseline = output['depth'][0].float().cpu().numpy()
        trt = torch.load(ROOT / f'result/optimization/reference-trt-scene-{scene}.pt',
                         map_location='cpu', weights_only=True)[0].numpy()
        assert baseline.shape == trt.shape == (480, 640)
        assert not np.isnan(baseline).any() and np.isfinite(trt).all() and (trt > 0).all()
        np.save(target / f'scene_{scene}_depth_level3.npy', baseline)
        np.save(target / f'scene_{scene}_depth_trt.npy', trt)
        print(f'Prepared scene {scene}', flush=True)


if __name__ == '__main__':
    main()
