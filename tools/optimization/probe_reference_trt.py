"""Benchmark the supplied ONNX FP16 reference engine, including GPU postprocessing.

Experimental only. The reference path disables depth-token masking and antialias
and omits the output mask; report these differences rather than replacing upstream.
"""
import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path
import torch
import tensorrt as trt
from probe_compile import ROOT, inputs, infer, measure, MDMModel
from mdm.utils.geo import depth_to_pointcloud


def reference_module():
    path = ROOT / 'result/optimization/reference-lingbot-depth-trt/tools/export_trt.py'
    spec = importlib.util.spec_from_file_location('reference_export', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stats(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    valid = torch.isfinite(a) & torch.isfinite(b) & (a > 0) & (b > 0)
    assert a.shape == b.shape == (1, 480, 640) and valid.any()
    delta = (a[valid] - b[valid]).abs()
    return {'mean_abs_m': delta.mean().item(), 'p99_abs_m': torch.quantile(delta, .99).item(),
            'max_abs_m': delta.max().item(),
            'mean_relative_pct': (delta / b[valid]).mean().item() * 100,
            'comparison_pixels_pct': valid.float().mean().item() * 100}


def time_call(call, repeats=50):
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000)
    return {'mean_ms': statistics.mean(values), 'median_ms': statistics.median(values),
            'p95_ms': sorted(values)[int(.95 * (len(values)-1))], 'runs': repeats}


def project(depth, intrinsics):
    depth = depth.float()
    return depth, depth_to_pointcloud(depth, intrinsics)


def main():
    data = [inputs(i) for i in range(8)]
    baseline_model = MDMModel.from_pretrained(ROOT / 'checkpoints/lingbot-depth-v0.5/model.pt').cuda().eval()
    baseline, standard_depth = [], []
    for i, frame in enumerate(data):
        out = infer(baseline_model, frame)
        standard_depth.append(out['depth'].float().cpu().clone())
        del out
        baseline.append({'scene': i, **measure(baseline_model, frame)})
    del baseline_model
    torch.cuda.empty_cache()
    print('BASELINE', json.dumps(baseline), flush=True)

    module = reference_module()
    wrapper = module.load_export_model(argparse.Namespace(
        model=str(ROOT / 'checkpoints/lingbot-depth-v0.5/model.pt'), precision='fp16', num_tokens=2000),
        torch.device('cuda'))
    half_data = [(im.half().contiguous(), d[None].half().contiguous(), k) for im, d, k in data]
    ref_depth = []
    with torch.inference_mode():
        for im, d, _ in half_data:
            ref_depth.append(wrapper(im, d).float().cpu().clone())
    del wrapper
    torch.cuda.empty_cache()

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    path = ROOT / 'result/optimization/reference-trt-nt2000-fp16/lingbot_depth_nt2000.engine'
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    assert engine is not None
    context = engine.create_execution_context()
    image_buffer = torch.empty(tuple(engine.get_tensor_shape('image')), dtype=torch.float16, device='cuda')
    depth_buffer = torch.empty(tuple(engine.get_tensor_shape('depth')), dtype=torch.float16, device='cuda')
    output = torch.empty(tuple(engine.get_tensor_shape('depth_refined')), dtype=torch.float16, device='cuda')
    for name, tensor in [('image', image_buffer), ('depth', depth_buffer), ('depth_refined', output)]:
        assert engine.get_tensor_dtype(name) == trt.DataType.HALF
        context.set_tensor_address(name, tensor.data_ptr())
    # TensorRT warns that the default stream introduces extra synchronization.
    # Keep engine execution and GPU postprocessing on the same dedicated stream.
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    def execute():
        assert context.execute_async_v3(stream.cuda_stream)
        return output

    # All inputs are GPU-resident; copies into stable engine buffers happen
    # before each scene's timing. Single-frame timing synchronizes every run.
    rows = []
    for i, (im, d, k) in enumerate(half_data):
        image_buffer.copy_(im); depth_buffer.copy_(d); torch.cuda.synchronize()
        execute(); torch.cuda.synchronize()
        assert torch.isfinite(output).all() and (output > 0).all()
        row = {'scene': i, 'engine_only': time_call(execute),
               'engine_and_projection': time_call(lambda: project(execute(), k)),
               'vs_reference_fp16': stats(output, ref_depth[i]),
               'vs_standard_level3': stats(output, standard_depth[i])}
        points = project(output, k)[1]
        assert points.shape == (1, 480, 640, 3)
        torch.testing.assert_close(points[..., 2], output.float(), rtol=0, atol=0)
        torch.save(output.float().cpu(), ROOT / f'result/optimization/reference-trt-scene-{i}.pt')
        rows.append(row)
        print('RESULT', json.dumps(row), flush=True)

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            assert context.execute_async_v3(capture_stream.cuda_stream)
    capture_stream.synchronize()
    with torch.cuda.graph(graph, stream=capture_stream):
        assert context.execute_async_v3(capture_stream.cuda_stream)
    torch.cuda.synchronize()
    compiled_project = torch.compile(project, mode='reduce-overhead', fullgraph=True, dynamic=False)
    combination = []
    for i, (im, d, k) in enumerate(half_data):
        image_buffer.copy_(im); depth_buffer.copy_(d); torch.cuda.synchronize()
        def combined():
            torch.compiler.cudagraph_mark_step_begin()
            graph.replay()
            return compiled_project(output, k)
        depth, points = combined(); torch.cuda.synchronize()
        torch.testing.assert_close(points[..., 2], depth, rtol=0, atol=0)
        original = torch.load(ROOT / f'result/optimization/reference-trt-scene-{i}.pt', weights_only=True)
        torch.testing.assert_close(depth.cpu(), original, rtol=0, atol=0)
        del depth, points
        row = {'scene': i, **time_call(combined)}
        combination.append(row)
        print('COMBINED_RESULT', json.dumps(row), flush=True)
    report = {'status': 'PASS', 'reference_repository': 'https://github.com/Ar-Ray-code/lingbot-depth-trt',
              'reference_commit': '60e69ab6ea5c65a2f19a22388a20d5969a8f556b',
              'engine': str(path.relative_to(ROOT)), 'tensorrt_version': trt.__version__,
              'num_tokens': 2000, 'equivalent_resolution_level': 3, 'batch_size': 1,
              'execution_stream': 'dedicated non-default CUDA stream',
              'input_and_output_resolution': [640, 480], 'precision': 'float16',
              'semantic_changes': ['depth-token masking disabled', 'RGB antialias disabled',
                                   'output-mask head omitted from engine; all returned depths unmasked'],
              'measurement': 'Single-frame synchronized host wall time; GPU-resident inputs. Loading, conversion, compilation, input preprocessing and transfer excluded.',
              'baseline': baseline, 'results': rows, 'cuda_graph_plus_compiled_projection': combination,
              'baseline_mean_ms': statistics.mean(r['mean_ms'] for r in baseline),
              'engine_only_mean_ms': statistics.mean(r['engine_only']['mean_ms'] for r in rows),
              'engine_and_projection_mean_ms': statistics.mean(r['engine_and_projection']['mean_ms'] for r in rows),
              'combined_mean_ms': statistics.mean(r['mean_ms'] for r in combination),
              'limitations': 'FP16 reference path changes model behavior; comparisons are consistency checks, not accuracy against ground truth. Missing output mask prevents identical validity filtering.'}
    dest = ROOT / 'result/optimization/reference-trt-benchmark.json'
    dest.write_text(json.dumps(report, indent=2) + '\n')
    print('REPORT', dest, flush=True)


if __name__ == '__main__':
    main()
