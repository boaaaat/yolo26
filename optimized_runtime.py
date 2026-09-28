"""Direct TensorRT runtime with persistent buffers and optional CUDA graphs."""

from pathlib import Path

import numpy as np
import torch

from optimized_engine import read_engine


class TensorRTRunner:
    def __init__(self, path: Path, height: int, width: int, gpu: int,
                 use_cuda_graph: bool = True, fused_preprocessing: bool = True):
        import tensorrt as trt

        torch.cuda.set_device(gpu)
        self.device = torch.device("cuda", gpu)
        self.metadata, payload = read_engine(path)
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(payload)
        if self.engine is None:
            raise RuntimeError(f"TensorRT could not deserialize {path}; remove this engine and its manifest to rebuild")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT could not allocate its execution context")
        self.stream = torch.cuda.Stream(device=self.device)
        self.done = torch.cuda.Event(blocking=True)
        self.graph = None
        self._fused = None
        self._pending = False
        tensors = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        inputs = [n for n in tensors if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outputs = [n for n in tensors if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected exactly one image input and one NMS-free detection output")
        input_name, output_name = inputs[0], outputs[0]
        input_shape = tuple(self.engine.get_tensor_shape(input_name))
        output_shape = tuple(self.engine.get_tensor_shape(output_name))
        if input_shape != (1, 3, height, width):
            raise ValueError(f"Engine input {input_shape} does not match capture geometry")
        if len(output_shape) != 3 or output_shape[0] != 1 or output_shape[2] != 6 or output_shape[1] <= 0:
            raise ValueError(f"Expected static [1, candidates, 6] xyxy/score/class output, got {output_shape}")
        dtype_map = {trt.float32: torch.float32, trt.float16: torch.float16}
        try:
            input_dtype = dtype_map[self.engine.get_tensor_dtype(input_name)]
            output_dtype = dtype_map[self.engine.get_tensor_dtype(output_name)]
        except KeyError as exc:
            raise ValueError("Only FP16/FP32 engine I/O is supported") from exc
        for name in tensors:
            if self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise ValueError(f"Expected device I/O for {name}")
            if self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
                raise ValueError(f"Expected linear engine I/O for {name}")
        self.raw = torch.empty((height, width, 3), dtype=torch.uint8, device=self.device)
        self.input = torch.empty(input_shape, dtype=input_dtype, device=self.device)
        self.output = torch.empty(output_shape, dtype=output_dtype, device=self.device)
        self.host_output = torch.empty(output_shape, dtype=output_dtype, pin_memory=True)
        self.output_view = self.host_output.numpy()[0]
        if not self.context.set_tensor_address(input_name, self.input.data_ptr()):
            raise RuntimeError("Could not bind TensorRT input")
        if not self.context.set_tensor_address(output_name, self.output.data_ptr()):
            raise RuntimeError("Could not bind TensorRT output")
        self.stream.wait_stream(torch.cuda.current_stream(self.device))

        # Startup initialization: all subsequent live frames reuse these allocations.
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            self.raw.zero_()
            if fused_preprocessing:
                try:
                    from optimized_preprocess import fused_preprocess
                    fused_preprocess(self.raw, self.input)
                    self.stream.synchronize()
                    self._fused = fused_preprocess
                except Exception as exc:
                    print(f"Fused preprocessing unavailable ({exc}); using PyTorch CUDA preprocessing.")
            for _ in range(3):
                self._gpu_work()
        self.stream.synchronize()
        if use_cuda_graph:
            try:
                graph = torch.cuda.CUDAGraph()
                with torch.inference_mode(), torch.cuda.graph(graph, stream=self.stream):
                    self._gpu_work()
                self.graph = graph
            except Exception as exc:
                self.stream.synchronize()
                print(f"CUDA graph capture unavailable ({exc}); using asynchronous TensorRT execution.")
        print(f"TensorRT ready: I/O {input_dtype}, CUDA graph {'on' if self.graph else 'off'}, "
              f"preprocessing {'fused' if self._fused else 'PyTorch'}.")

    def _gpu_work(self):
        if self._fused is not None:
            self._fused(self.raw, self.input)
        else:
            # Persistent destination; strided copies also cast to the engine I/O dtype.
            for channel in range(3):
                self.input[0, channel].copy_(self.raw[:, :, 2 - channel])
            self.input.mul_(1.0 / 255.0)
        if not self.context.execute_async_v3(stream_handle=self.stream.cuda_stream):
            raise RuntimeError("TensorRT enqueue failed")

    def submit(self, pinned_bgr: torch.Tensor) -> None:
        if self._pending:
            raise RuntimeError("Only one inference may be in flight")
        # An enqueue failure after H2D starts must still prevent source-buffer reuse.
        self._pending = True
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            self.raw.copy_(pinned_bgr, non_blocking=True)
            if self.graph is not None:
                self.graph.replay()
            else:
                self._gpu_work()
            self.host_output.copy_(self.output, non_blocking=True)
            self.done.record(self.stream)

    @property
    def pending(self) -> bool:
        return self._pending

    def finish(self) -> np.ndarray:
        if not self._pending:
            raise RuntimeError("No inference is pending")
        # One completion wait instead of synchronizing each preprocessing/model stage.
        self.done.synchronize()
        self._pending = False
        return self.output_view

    def close(self) -> None:
        self.stream.synchronize()
        self._pending = False
