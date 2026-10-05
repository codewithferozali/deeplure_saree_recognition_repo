"""
Preprocessing Pipeline for Saree Images.
Handles image loading, corruption validation, aspect-ratio preserving resizing,
motif preservation, and tensor conversion.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, Tuple
from PIL import Image, ImageOps, ImageFile

# Enable PIL to handle slightly truncated images without throwing exceptions
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


def validate_image(image_path: Union[str, Path]) -> bool:
    """
    Validates whether an image file exists, is non-empty, and can be decoded.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        True if the image is valid and readable, False otherwise.
    """
    path = Path(image_path)
    if not path.is_file() or path.stat().st_size == 0:
        return False

    try:
        with Image.open(path) as img:
            img.verify()
        # verify() closes the file, reopen to ensure decoding succeeds
        with Image.open(path) as img:
            img.load()
            if img.size[0] < 10 or img.size[1] < 10:
                return False
        return True
    except Exception as e:
        logger.warning(f"Invalid or corrupted image detected: {path} (Error: {e})")
        return False


def safe_load_image(image_path: Union[str, Path]) -> Optional[Image.Image]:
    """
    Safely loads an image file and converts it to 8-bit RGB.
    Returns None if the image is corrupted or missing.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        PIL.Image in 'RGB' mode, or None upon failure.
    """
    try:
        path = Path(image_path)
        if not path.exists():
            return None
        with Image.open(path) as img:
            # Handle EXIF orientation if present
            img = ImageOps.exif_transpose(img)
            return img.convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to load image from {image_path}: {e}")
        return None


def resize_and_pad(image: Image.Image, target_size: int = 224, fill_color: Tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    """
    Resizes an image preserving its original aspect ratio and pads the remaining
    space with neutral gray. This prevents non-uniform geometric distortion of motifs.
    
    Args:
        image: Source PIL Image.
        target_size: Desired square width and height (e.g. 224).
        fill_color: Background padding color (neutral RGB gray).
        
    Returns:
        Padded square PIL Image of size (target_size, target_size).
    """
    orig_w, orig_h = image.size
    scale = target_size / max(orig_w, orig_h)
    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))

    # Resize using high quality bicubic resampling
    resized_img = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

    # Pad to target_size x target_size
    padded_img = Image.new("RGB", (target_size, target_size), fill_color)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    padded_img.paste(resized_img, (paste_x, paste_y))

    return padded_img


def clean_dataset_paths(image_paths: list) -> list:
    """
    Filters a list of image file paths, dropping corrupted or unreadable files.
    """
    valid_paths = []
    corrupted_count = 0
    for p in image_paths:
        if validate_image(p):
            valid_paths.append(p)
        else:
            corrupted_count += 1
            
    if corrupted_count > 0:
        logger.info(f"Filtered out {corrupted_count} corrupted or unreadable images.")
    return valid_paths
