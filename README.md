# ComfyUI-WanVideoKsampler

An advanced custom node for ComfyUI that provides optimized access to **Wan2.1**, a state-of-the-art video foundation model suite. The WanVideoKsampler node features intelligent memory management to enable higher resolution outputs and longer video sequences, even on consumer-grade hardware.

![image](https://github.com/user-attachments/assets/ab28c245-9743-438d-9043-cb7953683258)


## About Wan2.1

**Wan2.1** is a comprehensive and open suite of video foundation models that pushes the boundaries of video generation with these key features:

* 🏆 **SOTA Performance**: Consistently outperforms existing open-source models and state-of-the-art commercial solutions across multiple benchmarks
* 💻 **Consumer-grade GPU Support**: The T2V-1.3B model requires only 8.19 GB VRAM, making it compatible with almost all consumer-grade GPUs
* 🎬 **Multiple Tasks**: Excels in Text-to-Video, Image-to-Video, Video Editing, Text-to-Image, and Video-to-Audio
* 📝 **Visual Text Generation**: First video model capable of generating both Chinese and English text with robust quality
* 🔄 **Powerful Video VAE**: Wan-VAE delivers exceptional efficiency, encoding and decoding 1080P videos of any length while preserving temporal information

## Overview

ComfyUI-WanVideoKsampler extends ComfyUI's capabilities by providing an optimized interface to Wan2.1 models. Its sophisticated memory management system pushes the boundaries of what's possible with consumer hardware, allowing for higher resolution outputs and longer video sequences than standard implementations.

## Key Features

- **Advanced Memory Management**:
  - Real-time memory usage tracking and optimization
  - Intelligent garbage collection to free unused resources
  - Preventive memory cleanup to avoid CUDA out-of-memory errors
  - Memory usage reporting for better troubleshooting
  - Enables processing higher resolutions and longer videos than standard samplers

- **Optimized for Wan2.1**:
  - Designed specifically to leverage the capabilities of Wan2.1 models
  - Compatible with Wan2.1's T2V-1.3B model on GPUs with as little as 8GB VRAM
  - Support for Wan-VAE's efficient video encoding/decoding

- **Performance Monitoring**:
  - Tracks processing time for each operation
  - Reports memory usage before and after processing
  - Provides per-frame processing metrics

- **Robust Error Handling**:
  - Graceful handling of memory-related errors
  - Detailed error reporting with suggested solutions
  - Aggressive memory cleanup during error recovery

## Installation

1. Clone this repository into your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/ShmuelRonen/ComfyUI-WanVideoKsampler.git
```

2. Install the required dependencies:
```bash
pip install psutil
```

3. Restart ComfyUI

## Usage

The WanVideoKsampler works like a standard KSampler node but is optimized for video processing with Wan2.1 models:

1. Connect a video latent (from Wan-VAE Encode or other source) to the `video_latents` input
2. Connect your Wan2.1 model, positive and negative prompts as usual
3. Configure the sampling parameters
4. Run your workflow to generate the processed video frames

### Sample Performance

With the T2V-1.3B model on an RTX 4090:
- Generate a 5-second 480P video in approximately 4 minutes (without optimization techniques like quantization)
- Performance comparable to some closed-source commercial models

## Parameters

- **model**: The Wan2.1 diffusion model to use
- **positive**: Positive conditioning/prompt
- **negative**: Negative conditioning/prompt
- **video_latents**: Input video frames in latent space
- **seed**: Random seed for reproducibility
- **steps**: Number of sampling steps
- **cfg**: Classifier-free guidance scale
- **sampler_name**: Sampling algorithm (Euler, Euler a, DDIM, etc.)
- **scheduler**: Scheduler type (Normal, Karras, etc.)
- **denoise**: Denoising strength (0.0-1.0)

## Memory Management Deep Dive

The node's sophisticated memory management system is what makes it possible to process high-resolution videos and longer sequences than standard implementations, even on consumer GPUs:

### Key Memory Features

1. **Real-time Memory Monitoring**:
   - Constantly tracks GPU/CPU memory usage
   - Provides visibility into memory consumption during processing
   - Warns users when approaching memory limits

2. **Intelligent Resource Management**:
   - Performs strategic garbage collection at optimal points
   - Frees CUDA cache when needed but not excessively
   - Uses aggressive cleanup methods when memory pressure is high

3. **Memory-Aware Error Recovery**:
   - Detects CUDA out-of-memory errors with specific handling
   - Performs emergency memory cleanup to recover from errors
   - Provides actionable feedback to users about memory constraints

4. **Performance Optimization**:
   - Reports detailed memory usage statistics for optimization
   - Tracks processing time to identify bottlenecks
   - Minimizes unnecessary tensor operations to reduce memory footprint

## Tips for High-Resolution and Long Videos

To get the most out of the memory management capabilities:

1. **Monitor the logs** for memory usage statistics to understand your system limits

2. **Adjust video resolution gradually** - the node will handle higher resolutions much better than standard KSamplers, but there are still limits

3. **For extremely long videos** - consider breaking the work into multiple segments and then concatenating them

4. **If processing 4K or higher resolution** - make sure you have adequate VRAM (16GB+ recommended, though the memory management will help push smaller GPUs further)

5. **Leverage Wan-VAE** - The Wan-VAE is designed to handle 1080P videos of any length, and the WanVideoKsampler is optimized to work with it

## Example Workflows

### Basic Text-to-Video:
```
Text Prompt -> CLIP Text Encode -> Wan2.1 T2V Model -> WanVideoKsampler -> Wan-VAE Decode -> SaveVideo
```

### Image-to-Video:
```
Image -> Wan2.1 I2V Setup -> WanVideoKsampler -> Wan-VAE Decode -> SaveVideo
```

For more complex examples and full workflows, check the [examples directory](examples/).

## Requirements

- ComfyUI (latest version recommended)
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (min 8GB VRAM for optimal performance)
- psutil library

## License

[MIT License](LICENSE)

## Acknowledgments

- This project is built to work with the outstanding Wan2.1 video foundation models
- Special thanks to the ComfyUI community and all contributors
