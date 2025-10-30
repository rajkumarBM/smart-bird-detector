#!/usr/bin/env python3
"""
Convert ONNX model to TensorRT engine for optimized inference.
This script converts Faster R-CNN or other ONNX models to TensorRT format.
"""

import argparse
import os
import tensorrt as trt


class TensorRTConverter:
    """Helper class to convert ONNX models to TensorRT engines."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)

    def build_engine(
        self,
        onnx_file_path,
        engine_file_path,
        precision="fp32",
        max_batch_size=1,
        max_workspace_size=1 << 30,  # 1GB
        min_shape=(1, 3, 480, 480),
        opt_shape=(1, 3, 640, 640),
        max_shape=(1, 3, 1280, 1280),
    ):
        """
        Build TensorRT engine from ONNX model.

        Args:
            onnx_file_path: Path to input ONNX model
            engine_file_path: Path to save TensorRT engine
            precision: Precision mode ('fp32', 'fp16', or 'int8')
            max_batch_size: Maximum batch size
            max_workspace_size: Maximum workspace size in bytes
            min_shape: Minimum input shape (batch, channels, height, width)
            opt_shape: Optimal input shape for best performance
            max_shape: Maximum input shape
        """

        if not os.path.exists(onnx_file_path):
            raise FileNotFoundError(f"ONNX file not found: {onnx_file_path}")

        print(f"Loading ONNX model: {onnx_file_path}")

        # Create builder and network
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)

        # Parse ONNX model
        print("Parsing ONNX model...")
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("Failed to parse ONNX model.")
                for error in range(parser.num_errors):
                    print(f"Error {error}: {parser.get_error(error)}")
                return None

        print(f"Successfully parsed ONNX model with {network.num_layers} layers")

        # Create builder config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)

        # Set precision mode
        if precision == "fp16":
            if builder.platform_has_fast_fp16:
                print("Enabling FP16 precision mode")
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                print("Warning: FP16 not supported on this platform, using FP32")
        elif precision == "int8":
            if builder.platform_has_fast_int8:
                print("Enabling INT8 precision mode")
                config.set_flag(trt.BuilderFlag.INT8)
                print("Warning: INT8 calibration not implemented, results may be suboptimal")
            else:
                print("Warning: INT8 not supported on this platform, using FP32")
        else:
            print("Using FP32 precision mode")

        # Configure dynamic shapes if needed
        profile = builder.create_optimization_profile()

        # Set dynamic shape for input tensor
        input_tensor = network.get_input(0)
        input_name = input_tensor.name

        print(f"Configuring dynamic shapes for input: {input_name}")
        print(f"  Min shape:  {min_shape}")
        print(f"  Opt shape:  {opt_shape}")
        print(f"  Max shape:  {max_shape}")

        profile.set_shape(
            input_name,
            min_shape,
            opt_shape,
            max_shape
        )

        config.add_optimization_profile(profile)

        # Build engine
        print("\nBuilding TensorRT engine...")
        print("This may take several minutes depending on model complexity...")

        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            print("Failed to build TensorRT engine")
            return None

        # Save engine to file
        print(f"\nSaving TensorRT engine to: {engine_file_path}")
        with open(engine_file_path, 'wb') as f:
            f.write(serialized_engine)

        file_size_mb = os.path.getsize(engine_file_path) / (1024 * 1024)
        print(f"Engine saved successfully! Size: {file_size_mb:.2f} MB")

        return engine_file_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX model to TensorRT engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion (FP32)
  python convert_to_tensorrt.py --onnx model.onnx --output model.trt

  # FP16 precision for faster inference
  python convert_to_tensorrt.py --onnx model.onnx --output model.trt --precision fp16

  # Custom input shapes
  python convert_to_tensorrt.py --onnx model.onnx --output model.trt \\
    --min-height 480 --min-width 480 \\
    --opt-height 640 --opt-width 640 \\
    --max-height 1280 --max-width 1280
        """
    )

    parser.add_argument(
        "--onnx",
        type=str,
        required=True,
        help="Path to input ONNX model"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output TensorRT engine file (.trt or .engine)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "int8"],
        default="fp32",
        help="Precision mode (default: fp32). fp16 is faster but slightly less accurate."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Maximum batch size (default: 1)"
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=1024,
        help="Maximum workspace size in MB (default: 1024)"
    )

    # Dynamic shape arguments
    parser.add_argument(
        "--min-height",
        type=int,
        default=480,
        help="Minimum input height (default: 480)"
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=480,
        help="Minimum input width (default: 480)"
    )
    parser.add_argument(
        "--opt-height",
        type=int,
        default=640,
        help="Optimal input height (default: 640)"
    )
    parser.add_argument(
        "--opt-width",
        type=int,
        default=640,
        help="Optimal input width (default: 640)"
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=1280,
        help="Maximum input height (default: 1280)"
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Maximum input width (default: 1280)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Create converter
    converter = TensorRTConverter(verbose=args.verbose)

    # Convert model
    try:
        converter.build_engine(
            onnx_file_path=args.onnx,
            engine_file_path=args.output,
            precision=args.precision,
            max_batch_size=args.batch_size,
            max_workspace_size=args.workspace * 1024 * 1024,  # Convert MB to bytes
            min_shape=(args.batch_size, 3, args.min_height, args.min_width),
            opt_shape=(args.batch_size, 3, args.opt_height, args.opt_width),
            max_shape=(args.batch_size, 3, args.max_height, args.max_width),
        )
        print("\nConversion completed successfully!")
    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
