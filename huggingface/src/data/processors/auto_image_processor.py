from transformers import AutoImageProcessor
from src.utils.data_utils import load_yaml_config

def get_auto_image_processor(processor_config):
    """
    Load a Image proceseor

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        processor (AutoImageProcessor)
        image processor that preprocess image info to DETR format

    Raises:
        FileNotFoundError: If the config file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    return AutoImageProcessor.from_pretrained(processor_config['path'])