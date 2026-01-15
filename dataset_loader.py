"""
Dataset Loader: Real-World Handwritten Character Recognition Data

Fetches and processes the EMNIST (Extended MNIST) dataset from HuggingFace,
which contains real handwritten English characters (A-Z, a-z, 0-9).

Features:
- Downloads EMNIST dataset automatically
- Converts 28x28 images to 8x8 format
- Supports uppercase letters (A-Z) and digits (0-9)
- Caches data for fast subsequent loads
- Handles data augmentation and preprocessing

Dataset: EMNIST (Extended MNIST)
Source: HuggingFace datasets
Characters: Letters (A-Z) and Digits (0-9)
Original Size: 28x28 grayscale
Target Size: 8x8 binary
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from PIL import Image


def resize_image_to_8x8(image: np.ndarray) -> np.ndarray:
    """
    Resize a 28x28 grayscale image to 8x8 binary format.

    Uses PIL for high-quality downsampling with antialiasing.

    Args:
        image: 28x28 numpy array (grayscale, 0-255)

    Returns:
        8x8 binary numpy array (0 or 1)
    """
    # Convert to PIL Image
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    pil_img = Image.fromarray(image, mode='L')

    # Resize with high-quality antialiasing
    pil_img_resized = pil_img.resize((8, 8), Image.Resampling.LANCZOS)

    # Convert back to numpy
    resized = np.array(pil_img_resized)

    # Threshold to binary (using Otsu-like adaptive threshold)
    threshold = np.mean(resized)
    binary = (resized > threshold).astype(np.float32)

    return binary


def load_emnist_dataset(
    split: str = 'train',
    characters: Optional[List[str]] = None,
    max_samples_per_class: int = 1000,
    cache_dir: str = './dataset_cache'
) -> Dict[str, np.ndarray]:
    """
    Load EMNIST dataset from HuggingFace and convert to 8x8 format.

    Args:
        split: 'train' or 'test'
        characters: List of characters to load (e.g., ['A', 'B', 'C', '0', '1'])
                   If None, loads all uppercase letters and digits
        max_samples_per_class: Maximum samples per character class
        cache_dir: Directory to cache processed data

    Returns:
        Dictionary with:
        - 'inputs': (N, 64) array of flattened 8x8 bitmaps
        - 'labels': (N,) array of integer labels (0=A, 1=B, ..., 25=Z, 26=0, ..., 35=9)
        - 'char_map': Dictionary mapping label_idx -> character
        - 'label_map': Dictionary mapping character -> label_idx
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install the 'datasets' library: "
            "pip install datasets huggingface_hub"
        )

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)

    # Define character mapping (EMNIST ByClass format)
    # EMNIST labels: 0-9 = digits, 10-35 = uppercase A-Z, 36-61 = lowercase a-z
    # We'll focus on uppercase and digits for this implementation

    if characters is None:
        # Default: All uppercase letters and digits
        characters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    # Create label mapping
    label_map = {char: idx for idx, char in enumerate(characters)}
    char_map = {idx: char for char, idx in label_map.items()}

    print(f"[*] Loading EMNIST dataset ({split} split)...")
    print(f"[*] Target characters: {characters}")

    # Try to load cached data
    cache_file = os.path.join(
        cache_dir,
        f"emnist_{split}_{''.join(characters)}_{max_samples_per_class}.npz"
    )

    if os.path.exists(cache_file):
        print(f"[*] Loading from cache: {cache_file}")
        data = np.load(cache_file)
        return {
            'inputs': data['inputs'],
            'labels': data['labels'],
            'char_map': char_map,
            'label_map': label_map
        }

    # Load EMNIST dataset from HuggingFace
    try:
        print("[*] Downloading EMNIST dataset from HuggingFace...")
        print("[*] This may take a few minutes on first run...")

        # Load EMNIST ByClass (has all characters separated)
        dataset = load_dataset('emnist', 'byclass', split=split)

        print(f"[*] Dataset loaded: {len(dataset)} total samples")

    except Exception as e:
        print(f"[!] Error loading EMNIST: {e}")
        print("[*] Falling back to synthetic data generation...")
        return _generate_synthetic_dataset(characters, max_samples_per_class)

    # Convert EMNIST labels to our character set
    # EMNIST ByClass mapping:
    # 0-9: digits 0-9
    # 10-35: uppercase A-Z
    # 36-61: lowercase a-z

    def emnist_label_to_char(label: int) -> Optional[str]:
        """Convert EMNIST label to character."""
        if 0 <= label <= 9:
            return str(label)
        elif 10 <= label <= 35:
            return chr(ord('A') + (label - 10))
        elif 36 <= label <= 61:
            return chr(ord('a') + (label - 36))
        return None

    # Collect samples for each character
    inputs_list = []
    labels_list = []

    samples_per_class = {char: 0 for char in characters}

    print("[*] Processing and resizing images to 8x8...")

    for idx, sample in enumerate(dataset):
        if idx % 5000 == 0:
            print(f"    Processed {idx}/{len(dataset)} samples...")

        # Get EMNIST label and convert to character
        emnist_label = sample['label']
        char = emnist_label_to_char(emnist_label)

        # Skip if not in our target character set
        if char not in characters:
            continue

        # Skip if we have enough samples for this character
        if samples_per_class[char] >= max_samples_per_class:
            continue

        # Get image (28x28)
        image = np.array(sample['image'])

        # Resize to 8x8
        image_8x8 = resize_image_to_8x8(image)

        # Flatten to 64-dimensional vector
        flat_image = image_8x8.flatten()

        # Store
        inputs_list.append(flat_image)
        labels_list.append(label_map[char])
        samples_per_class[char] += 1

        # Check if we have enough samples for all characters
        if all(count >= max_samples_per_class for count in samples_per_class.values()):
            print(f"[*] Collected {max_samples_per_class} samples for all characters!")
            break

    # Convert to numpy arrays
    inputs = np.array(inputs_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)

    print(f"[*] Dataset ready: {len(inputs)} samples")
    print(f"[*] Samples per character: {samples_per_class}")

    # Cache the processed data
    print(f"[*] Caching to: {cache_file}")
    np.savez_compressed(
        cache_file,
        inputs=inputs,
        labels=labels
    )

    return {
        'inputs': inputs,
        'labels': labels,
        'char_map': char_map,
        'label_map': label_map
    }


def _generate_synthetic_dataset(
    characters: List[str],
    max_samples_per_class: int
) -> Dict[str, np.ndarray]:
    """
    Fallback: Generate synthetic dataset using existing data_factory.

    Args:
        characters: List of characters
        max_samples_per_class: Samples per character

    Returns:
        Dataset dictionary
    """
    from data_factory import get_character_bitmap

    print("[*] Generating synthetic dataset...")

    inputs_list = []
    labels_list = []

    label_map = {char: idx for idx, char in enumerate(characters)}
    char_map = {idx: char for char, idx in label_map.items()}

    for char in characters:
        for _ in range(max_samples_per_class):
            # Get base template
            try:
                bitmap = get_character_bitmap(char)
            except:
                # If character not available, skip
                continue

            # Add noise
            noise_level = np.random.uniform(0.05, 0.15)
            noisy_bitmap = bitmap.copy()
            flip_mask = np.random.rand(64) < noise_level
            noisy_bitmap[flip_mask] = 1 - noisy_bitmap[flip_mask]

            inputs_list.append(noisy_bitmap)
            labels_list.append(label_map[char])

    inputs = np.array(inputs_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)

    print(f"[*] Generated {len(inputs)} synthetic samples")

    return {
        'inputs': inputs,
        'labels': labels,
        'char_map': char_map,
        'label_map': label_map
    }


def augment_dataset(
    inputs: np.ndarray,
    labels: np.ndarray,
    noise_level: float = 0.1,
    shift_range: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment dataset with noise and shifts.

    Args:
        inputs: (N, 64) input array
        labels: (N,) label array
        noise_level: Probability of flipping each bit
        shift_range: Maximum pixel shift (not implemented for 8x8)

    Returns:
        Augmented (inputs, labels)
    """
    augmented_inputs = inputs.copy()

    if noise_level > 0:
        # Add random bit flips
        flip_mask = np.random.rand(*augmented_inputs.shape) < noise_level
        augmented_inputs[flip_mask] = 1 - augmented_inputs[flip_mask]

    return augmented_inputs, labels


def visualize_samples(
    inputs: np.ndarray,
    labels: np.ndarray,
    char_map: Dict[int, str],
    num_samples: int = 10
):
    """
    Visualize random samples from the dataset.

    Args:
        inputs: (N, 64) input array
        labels: (N,) label array
        char_map: Mapping from label_idx to character
        num_samples: Number of samples to show
    """
    import matplotlib.pyplot as plt

    # Select random samples
    indices = np.random.choice(len(inputs), min(num_samples, len(inputs)), replace=False)

    # Create grid
    cols = 5
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if num_samples > 1 else [axes]

    for i, idx in enumerate(indices):
        if i >= len(axes):
            break

        # Get sample
        bitmap = inputs[idx].reshape(8, 8)
        label = labels[idx]
        char = char_map[label]

        # Plot
        axes[i].imshow(bitmap, cmap='gray', interpolation='nearest')
        axes[i].set_title(f"'{char}' (Label {label})")
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == '__main__':
    print("=== EMNIST Dataset Loader Test ===\n")

    # Load a small subset for testing
    test_chars = ['A', 'B', 'C', 'X', 'Y', 'Z', '0', '1', '2']

    dataset = load_emnist_dataset(
        split='train',
        characters=test_chars,
        max_samples_per_class=100
    )

    print("\n=== Dataset Info ===")
    print(f"Inputs shape: {dataset['inputs'].shape}")
    print(f"Labels shape: {dataset['labels'].shape}")
    print(f"Character map: {dataset['char_map']}")
    print(f"Label map: {dataset['label_map']}")

    # Visualize some samples
    print("\n[*] Visualizing samples...")
    visualize_samples(
        dataset['inputs'],
        dataset['labels'],
        dataset['char_map'],
        num_samples=9
    )

    print("\n✅ Test complete!")

