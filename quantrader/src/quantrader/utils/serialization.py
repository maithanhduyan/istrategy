"""
Utility functions for serialization, validation, and common operations.
"""

import json
import pickle
import gzip
import hashlib
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Union, Type
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SerializationUtils:
    """Utilities for serializing and deserializing objects."""

    @staticmethod
    def save_json(
        data: Any, filepath: Union[str, Path], compressed: bool = False
    ) -> None:
        """Save data as JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        json_str = json.dumps(
            data, default=SerializationUtils._json_serializer, indent=2
        )

        if compressed:
            with gzip.open(f"{filepath}.gz", "wt", encoding="utf-8") as f:
                f.write(json_str)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)

        logger.info(f"Saved JSON to {filepath}")

    @staticmethod
    def load_json(filepath: Union[str, Path], compressed: bool = False) -> Any:
        """Load data from JSON file."""
        filepath = Path(filepath)

        if compressed:
            with gzip.open(f"{filepath}.gz", "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

        logger.info(f"Loaded JSON from {filepath}")
        return data

    @staticmethod
    def save_pickle(
        data: Any, filepath: Union[str, Path], compressed: bool = True
    ) -> None:
        """Save data as pickle file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if compressed:
            with gzip.open(f"{filepath}.gz", "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Saved pickle to {filepath}")

    @staticmethod
    def load_pickle(filepath: Union[str, Path], compressed: bool = True) -> Any:
        """Load data from pickle file."""
        filepath = Path(filepath)

        if compressed:
            with gzip.open(f"{filepath}.gz", "rb") as f:
                data = pickle.load(f)
        else:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

        logger.info(f"Loaded pickle from {filepath}")
        return data

    @staticmethod
    def save_torch_model(
        model: torch.nn.Module,
        filepath: Union[str, Path],
        include_architecture: bool = True,
    ) -> None:
        """Save PyTorch model with optional architecture info."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
            "timestamp": datetime.now().isoformat(),
        }

        if include_architecture and hasattr(model, "get_architecture_dict"):
            save_dict["architecture"] = model.get_architecture_dict()

        torch.save(save_dict, filepath)
        logger.info(f"Saved PyTorch model to {filepath}")

    @staticmethod
    def load_torch_model(
        filepath: Union[str, Path], model_class: Optional[Type] = None
    ) -> Dict[str, Any]:
        """Load PyTorch model."""
        filepath = Path(filepath)

        checkpoint = torch.load(filepath, map_location="cpu")
        logger.info(f"Loaded PyTorch model from {filepath}")

        return checkpoint

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Custom JSON serializer for numpy arrays and other objects."""
        if isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": obj.tolist(),
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, datetime):
            return {"__datetime__": True, "isoformat": obj.isoformat()}
        elif hasattr(obj, "to_dict"):
            return obj.to_dict()
        else:
            return str(obj)


class ValidationUtils:
    """Utilities for validating data and models."""

    @staticmethod
    def validate_genome_integrity(genome_dict: Dict[str, Any]) -> bool:
        """Validate genome data integrity."""
        required_fields = ["genome_id", "architecture", "weights", "biases"]

        for field in required_fields:
            if field not in genome_dict:
                logger.error(f"Missing required field: {field}")
                return False

        # Validate architecture
        arch = genome_dict["architecture"]
        if not isinstance(arch, dict) or "layers" not in arch:
            logger.error("Invalid architecture format")
            return False

        # Validate weights and biases are dictionaries
        if not isinstance(genome_dict["weights"], dict):
            logger.error("Weights must be a dictionary")
            return False

        if not isinstance(genome_dict["biases"], dict):
            logger.error("Biases must be a dictionary")
            return False

        logger.info("Genome validation passed")
        return True

    @staticmethod
    def validate_market_data(data: Dict[str, Any]) -> bool:
        """Validate market data format."""
        required_fields = [
            "symbol",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return False

        # Validate price consistency
        try:
            high = float(data["high"])
            low = float(data["low"])
            open_price = float(data["open"])
            close = float(data["close"])

            if not (low <= open_price <= high and low <= close <= high):
                logger.error("Price data inconsistency")
                return False

        except (ValueError, TypeError):
            logger.error("Invalid price data types")
            return False

        logger.info("Market data validation passed")
        return True

    @staticmethod
    def validate_neural_network(model: torch.nn.Module, input_shape: tuple) -> bool:
        """Validate neural network can process given input shape."""
        try:
            model.eval()
            dummy_input = torch.randn(*input_shape)

            with torch.no_grad():
                output = model(dummy_input)

            if output is None:
                logger.error("Model returned None")
                return False

            logger.info(
                f"Model validation passed - input: {input_shape}, output: {output.shape}"
            )
            return True

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False


class HashUtils:
    """Utilities for creating reproducible hashes."""

    @staticmethod
    def hash_dict(data: Dict[str, Any], algorithm: str = "md5") -> str:
        """Create deterministic hash of dictionary."""
        # Convert to deterministic JSON string
        json_str = json.dumps(data, sort_keys=True, default=str)

        # Create hash
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(json_str.encode("utf-8"))

        return hash_obj.hexdigest()

    @staticmethod
    def hash_array(array: np.ndarray, algorithm: str = "md5") -> str:
        """Create hash of numpy array."""
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(array.tobytes())

        return hash_obj.hexdigest()

    @staticmethod
    def hash_model_weights(model: torch.nn.Module, algorithm: str = "md5") -> str:
        """Create hash of model weights."""
        hash_obj = hashlib.new(algorithm)

        for name, param in sorted(model.named_parameters()):
            hash_obj.update(name.encode("utf-8"))
            hash_obj.update(param.detach().cpu().numpy().tobytes())

        return hash_obj.hexdigest()


class PathUtils:
    """Utilities for path management."""

    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """Ensure directory exists, create if not."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def get_project_root() -> Path:
        """Get project root directory."""
        current = Path(__file__).parent
        while current.parent != current:
            if (current / "pyproject.toml").exists():
                return current
            current = current.parent
        return Path.cwd()

    @staticmethod
    def get_data_dir() -> Path:
        """Get data directory."""
        return PathUtils.ensure_directory(PathUtils.get_project_root() / "data")

    @staticmethod
    def get_models_dir() -> Path:
        """Get models directory."""
        return PathUtils.ensure_directory(PathUtils.get_project_root() / "models")

    @staticmethod
    def get_logs_dir() -> Path:
        """Get logs directory."""
        return PathUtils.ensure_directory(PathUtils.get_project_root() / "logs")


class ConfigUtils:
    """Utilities for configuration management."""

    @staticmethod
    def load_config(filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file."""
        filepath = Path(filepath)

        if filepath.suffix == ".json":
            return SerializationUtils.load_json(filepath)
        elif filepath.suffix in [".yaml", ".yml"]:
            try:
                import yaml

                with open(filepath, "r") as f:
                    return yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required for YAML config files")
        else:
            raise ValueError(f"Unsupported config file format: {filepath.suffix}")

    @staticmethod
    def save_config(config: Dict[str, Any], filepath: Union[str, Path]) -> None:
        """Save configuration to file."""
        filepath = Path(filepath)

        if filepath.suffix == ".json":
            SerializationUtils.save_json(config, filepath)
        elif filepath.suffix in [".yaml", ".yml"]:
            try:
                import yaml

                with open(filepath, "w") as f:
                    yaml.safe_dump(config, f, default_flow_style=False)
            except ImportError:
                raise ImportError("PyYAML is required for YAML config files")
        else:
            raise ValueError(f"Unsupported config file format: {filepath.suffix}")

    @staticmethod
    def merge_configs(
        base_config: Dict[str, Any], override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge two configuration dictionaries."""
        merged = base_config.copy()

        def _merge_recursive(base: Dict, override: Dict) -> Dict:
            for key, value in override.items():
                if (
                    key in base
                    and isinstance(base[key], dict)
                    and isinstance(value, dict)
                ):
                    base[key] = _merge_recursive(base[key], value)
                else:
                    base[key] = value
            return base

        return _merge_recursive(merged, override_config)


class ArrayUtils:
    """Utilities for array operations."""

    @staticmethod
    def normalize_array(arr: np.ndarray, method: str = "minmax") -> np.ndarray:
        """Normalize array using specified method."""
        if method == "minmax":
            return (arr - arr.min()) / (arr.max() - arr.min())
        elif method == "zscore":
            return (arr - arr.mean()) / arr.std()
        elif method == "robust":
            median = np.median(arr)
            mad = np.median(np.abs(arr - median))
            return (arr - median) / mad
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    @staticmethod
    def sliding_window(arr: np.ndarray, window_size: int, step: int = 1) -> np.ndarray:
        """Create sliding windows from array."""
        shape = ((arr.shape[0] - window_size) // step + 1, window_size) + arr.shape[1:]
        strides = (arr.strides[0] * step,) + arr.strides
        return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    @staticmethod
    def downsample_array(
        arr: np.ndarray, factor: int, method: str = "mean"
    ) -> np.ndarray:
        """Downsample array by factor."""
        if method == "mean":
            return arr[::factor]
        elif method == "max":
            return np.array(
                [arr[i : i + factor].max() for i in range(0, len(arr), factor)]
            )
        elif method == "min":
            return np.array(
                [arr[i : i + factor].min() for i in range(0, len(arr), factor)]
            )
        else:
            raise ValueError(f"Unknown downsampling method: {method}")


class TimingUtils:
    """Utilities for performance timing."""

    def __init__(self):
        self.timers = {}

    def start_timer(self, name: str) -> None:
        """Start a timer."""
        self.timers[name] = {"start": datetime.now(), "end": None, "duration": None}

    def end_timer(self, name: str) -> float:
        """End a timer and return duration in seconds."""
        if name not in self.timers:
            raise ValueError(f"Timer '{name}' not found")

        self.timers[name]["end"] = datetime.now()
        duration = (
            self.timers[name]["end"] - self.timers[name]["start"]
        ).total_seconds()
        self.timers[name]["duration"] = duration

        return duration

    def get_timer_summary(self) -> Dict[str, float]:
        """Get summary of all timers."""
        return {
            name: timer["duration"]
            for name, timer in self.timers.items()
            if timer["duration"] is not None
        }


# Global timer instance
timer = TimingUtils()
