import os
import numpy as np
import torch
from airsoul.utils import log_debug, log_fatal
from airsoul.utils.simple_generator import ModelLoader
from airsoul.models import OmniRL
import yaml
from airsoul.utils import Configure

def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        log_fatal(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Configure(config_dict)

def main(config_path):
    # Load configuration
    config = load_config(config_path)
    log_debug(f"Loaded config from {config_path}")

    # Initialize ModelLoader
    try:
        loader = ModelLoader(config=config, use_gpu=True, world_size=1)
        log_debug("ModelLoader initialized")
    except Exception as e:
        log_fatal(f"Failed to initialize ModelLoader: {e}")
        raise

    # Load model
    try:
        model = loader.load_model(OmniRL)
        log_debug("Model loaded successfully")
    except Exception as e:
        log_fatal(f"Failed to load model: {e}")
        raise

    # Prepare input for generate
    state_input = np.zeros(48, dtype=np.float32)  # Match state_encode.input_size
    interactive_prompt = np.array([0.0, 0.5, 0.0], dtype=np.float32)
    interactive_tag = np.array([4], dtype=np.int64)

    # Test generate interface
    with torch.no_grad():
        state_dist, action, reward = model.module.generate(
            observation=state_input,
            prompt=interactive_prompt,
            tag=interactive_tag,
            temp=1.0,
            future_prediction=False
        )
    log_debug("Generate interface test completed")
    log_debug("Generate interface test completed")
    print("Generate Interface Test Result:")
    print(f"State Distribution: {state_dist}")
    print(f"Action: {action}")
    reward = np.array([0.0], dtype=np.float32)
    print(f"Reward: {reward}")

    # Prepare inputs for in_context_learn
    # action_input = np.zeros(12, dtype=np.float32)  # Placeholder, adjust based on action_encode.input_size
    # reward_input = np.array([0.0], dtype=np.float32)  # Placeholder, adjust based on reward_encode.input_size

    # Test in_context_learn interface
    try:
        with torch.no_grad():
            new_cache = model.module.in_context_learn(
                observation=state_input,
                prompts=interactive_prompt,
                tags=interactive_tag,
                action=action,
                reward=reward,
                need_cache=True,
                single_batch=True,
                single_step=True
            )
        log_debug("In-context learn interface test completed")
        print("In-Context Learn Interface Test Result:")
        print(f"New Cache: {new_cache.shape if isinstance(new_cache, torch.Tensor) else new_cache}")
    except Exception as e:
        log_fatal(f"In-context learn interface test failed: {e}")
        raise

if __name__ == "__main__":
    config_path = "config_test.yaml"
    main(config_path)
