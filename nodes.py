import torch
import comfy.sample
import comfy.model_management
import comfy.utils
import gc
import logging
import nodes
from typing import Dict, Union
import time
from contextlib import contextmanager
import psutil


class MemoryManager:
    """Manages memory resources for efficient video processing."""
    
    def __init__(self, device=None, log_level: str = "INFO"):
        self.logger = logging.getLogger("MemoryManager")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(getattr(logging, log_level))
        
        self.device = device or comfy.model_management.get_torch_device()
        self.logger.info(f"Using device: {self.device}")
        
        # Memory thresholds (percentages)
        self.warning_threshold = 85
        self.critical_threshold = 95
    
    def is_cuda_device(self) -> bool:
        """Check if the current device is a CUDA device."""
        if isinstance(self.device, str):
            return self.device.startswith("cuda")
        elif isinstance(self.device, torch.device):
            return self.device.type == "cuda"
        return False
    
    def get_memory_stats(self) -> Dict[str, Union[int, float]]:
        """Get current memory statistics for the device."""
        stats = {}
        
        if self.is_cuda_device() and torch.cuda.is_available():
            try:
                t = torch.cuda.get_device_properties(0)
                stats["total"] = t.total_memory
                stats["reserved"] = torch.cuda.memory_reserved(0)
                stats["allocated"] = torch.cuda.memory_allocated(0)
                stats["free"] = stats["total"] - stats["reserved"]
                stats["usage_percent"] = (stats["allocated"] / stats["total"]) * 100
            except Exception as e:
                self.logger.error(f"Error getting CUDA memory stats: {e}")
                stats = {"error": str(e)}
        else:
            # CPU memory stats
            vm = psutil.virtual_memory()
            stats["total"] = vm.total
            stats["available"] = vm.available
            stats["used"] = vm.used
            stats["free"] = vm.free
            stats["usage_percent"] = vm.percent
            
        return stats
    
    def is_memory_critical(self) -> bool:
        """Check if memory usage is at critical levels."""
        stats = self.get_memory_stats()
        if "error" in stats:
            return True  # Assume critical if we can't get stats
        
        return stats.get("usage_percent", 0) > self.critical_threshold
    
    @contextmanager
    def track_memory(self, label: str = "Operation"):
        """Context manager to track memory usage before and after an operation."""
        if self.is_cuda_device() and torch.cuda.is_available():
            start_mem = torch.cuda.memory_allocated()
            start_time = time.time()
            try:
                yield
            finally:
                end_mem = torch.cuda.memory_allocated()
                end_time = time.time()
                self.logger.info(f"{label} - Memory change: {(end_mem-start_mem)/1024**2:.2f}MB, Time: {end_time-start_time:.2f}s")
        else:
            start_time = time.time()
            try:
                yield
            finally:
                end_time = time.time()
                self.logger.info(f"{label} - Time: {end_time-start_time:.2f}s")
    
    def cleanup(self, force: bool = False):
        """Clean up memory resources."""
        if self.is_cuda_device() and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        if force:
            # More aggressive cleanup
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) and not obj.is_cuda:
                        del obj
                except:
                    pass
            gc.collect()
            if self.is_cuda_device() and torch.cuda.is_available():
                torch.cuda.empty_cache()


class WanVideoKsampler:
    """
    Video K-sampler node with memory management for processing video latents.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "video_latents": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    
    def __init__(self):
        self.logger = logging.getLogger("WanVideoKsampler")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Initialize memory manager
        self.memory_manager = None

    def sample(
        self, 
        model,
        video_latents: Dict[str, torch.Tensor], 
        positive,
        negative,
        seed: int, 
        steps: int,
        cfg: float, 
        sampler_name: str, 
        scheduler: str, 
        denoise: float
    ) -> Dict[str, torch.Tensor]:
        """
        Sample video frames with memory management.
        
        Args:
            model: Diffusion model
            video_latents: Dictionary containing latent tensors
            positive: Positive conditioning
            negative: Negative conditioning
            seed: Random seed
            steps: Number of sampling steps
            cfg: Classifier-free guidance scale
            sampler_name: Name of sampler to use
            scheduler: Name of scheduler to use
            denoise: Denoising strength
            
        Returns:
            Dictionary containing processed latent tensors
        """
        start_time = time.time()
        device = comfy.model_management.get_torch_device()
        
        # Initialize memory manager if needed
        if self.memory_manager is None:
            self.memory_manager = MemoryManager(device)
        
        # Log latent size for debugging
        if isinstance(video_latents, dict) and 'samples' in video_latents:
            latent_samples = video_latents['samples']
            total_frames = latent_samples.shape[0]
            self.logger.info(f"Processing latent shape: {latent_samples.shape}, total frames: {total_frames}")
        else:
            self.logger.error("Invalid latent format")
            raise ValueError("Expected latent dictionary with 'samples' key")
        
        self.logger.info(f"Processing with {steps} steps, {cfg} CFG, {sampler_name} sampler")
        
        try:
            # Process with memory tracking
            with self.memory_manager.track_memory("Video processing"):
                # Check memory usage before processing
                memory_stats = self.memory_manager.get_memory_stats()
                if "usage_percent" in memory_stats:
                    self.logger.info(f"Memory usage before processing: {memory_stats['usage_percent']:.1f}%")
                
                # Apply sampling
                result = nodes.common_ksampler(
                    model, seed, steps, cfg, sampler_name, scheduler, 
                    positive, negative, video_latents, denoise=denoise
                )
                
                # Clear memory after processing
                self.memory_manager.cleanup()
                
                # Check memory usage after processing
                memory_stats = self.memory_manager.get_memory_stats()
                if "usage_percent" in memory_stats:
                    self.logger.info(f"Memory usage after processing: {memory_stats['usage_percent']:.1f}%")
                
                end_time = time.time()
                self.logger.info(f"Complete: {total_frames} frames in {end_time - start_time:.2f}s ({(end_time - start_time) / total_frames:.2f}s per frame)")
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error during processing: {str(e)}")
            # Try to release memory
            self.memory_manager.cleanup(force=True)
            # Check if it's an out-of-memory error
            if "CUDA out of memory" in str(e):
                self.logger.error("Out of memory error. Consider reducing frame count or model complexity.")
            raise e


# Node registration
NODE_CLASS_MAPPINGS = {
    "WanVideoKsampler": WanVideoKsampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoKsampler": "Wan Video Ksampler",
}