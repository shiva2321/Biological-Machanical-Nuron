"""
Data Factory: Procedural Training Data Generator for SNNs

Generates infinite variations of 8x8 character bitmaps (A-Z, 0-9) with:
- Perfect character templates
- Random shifts (translation)
- Random noise (bit flips)
- On-the-fly augmentation

No static files needed - generates training data procedurally!
"""

import numpy as np
from typing import Dict, List


# ============================================================================
# Character Bitmap Templates (8x8)
# ============================================================================

# Hardcoded perfect 8x8 binary patterns for all characters
CHARACTER_BITMAPS = {
    # Letters A-Z
    'A': [
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
    ],
    'B': [
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
    ],
    'C': [
        [0, 0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 1, 1, 0],
    ],
    'D': [
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
    ],
    'E': [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    'F': [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
    ],
    'G': [
        [0, 0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 1, 1, 0],
    ],
    'H': [
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
    ],
    'I': [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    'J': [
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
    ],
    'K': [
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
    ],
    'L': [
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    'M': [
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
    ],
    'N': [
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 1, 1, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
    ],
    'O': [
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
    ],
    'P': [
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
    ],
    'Q': [
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1],
    ],
    'R': [
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 1],
    ],
    'S': [
        [0, 0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
    ],
    'T': [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
    ],
    'U': [
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
    ],
    'V': [
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
    ],
    'W': [
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
    ],
    'X': [
        [1, 1, 0, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 1],
    ],
    'Y': [
        [1, 1, 0, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
    ],
    'Z': [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],

    # Digits 0-9
    '0': [
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 0, 0, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 1],
    ],
    '1': [
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    '2': [
        [0, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    '3': [
        [0, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
    ],
    '4': [
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
    ],
    '5': [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
    ],
    '6': [
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
    ],
    '7': [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
    ],
    '8': [
        [0, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
    ],
    '9': [
        [0, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
    ],
}


# ============================================================================
# Core Functions
# ============================================================================

def get_character_bitmap(char: str) -> np.ndarray:
    """
    Get the perfect 8x8 binary bitmap for a character.

    Args:
        char: Single character (A-Z or 0-9), case-insensitive

    Returns:
        Flattened 64-element numpy array (0s and 1s)

    Raises:
        ValueError: If character is not supported

    Example:
        >>> bitmap = get_character_bitmap('A')
        >>> bitmap.shape
        (64,)
        >>> bitmap[:8]  # First row
        array([0, 0, 1, 1, 1, 1, 0, 0])
    """
    char = char.upper()

    if char not in CHARACTER_BITMAPS:
        supported = ', '.join(sorted(CHARACTER_BITMAPS.keys()))
        raise ValueError(f"Character '{char}' not supported. Supported: {supported}")

    # Get 8x8 grid and flatten to 64 elements
    grid = np.array(CHARACTER_BITMAPS[char], dtype=np.float32)
    return grid.flatten()


def augment_data(base_grid: np.ndarray,
                 noise_prob: float = None,
                 shift_prob: float = 0.5) -> np.ndarray:
    """
    Apply realistic "hand-drawn" augmentation to a character bitmap.

    Augmentations:
    - Random shift (up/down/left/right by 1 pixel)
    - Random noise (bit flips)

    Args:
        base_grid: Flattened 64-element array (8x8 bitmap)
        noise_prob: Probability of flipping each bit (0.05-0.15 random if None)
        shift_prob: Probability of applying a shift (default 0.5)

    Returns:
        Augmented 64-element array

    Example:
        >>> original = get_character_bitmap('A')
        >>> noisy = augment_data(original)
        >>> # Should look similar but with some variations
    """
    # Reshape to 8x8
    grid = base_grid.reshape(8, 8).copy()

    # Apply random shift
    if np.random.rand() < shift_prob:
        shift_direction = np.random.choice(['up', 'down', 'left', 'right'])
        grid = _shift_grid(grid, shift_direction)

    # Apply noise
    if noise_prob is None:
        # Random noise level between 5% and 15%
        noise_prob = np.random.uniform(0.05, 0.15)

    # Flip random bits
    noise_mask = np.random.rand(8, 8) < noise_prob
    grid[noise_mask] = 1 - grid[noise_mask]

    # Ensure values are still binary (clip to [0, 1])
    grid = np.clip(grid, 0, 1)

    return grid.flatten()


def _shift_grid(grid: np.ndarray, direction: str) -> np.ndarray:
    """
    Shift an 8x8 grid by 1 pixel in the specified direction.

    Uses zero-padding (empty space filled with zeros).

    Args:
        grid: 8x8 numpy array
        direction: 'up', 'down', 'left', or 'right'

    Returns:
        Shifted 8x8 array
    """
    shifted = np.zeros_like(grid)

    if direction == 'up':
        shifted[:-1, :] = grid[1:, :]
    elif direction == 'down':
        shifted[1:, :] = grid[:-1, :]
    elif direction == 'left':
        shifted[:, :-1] = grid[:, 1:]
    elif direction == 'right':
        shifted[:, 1:] = grid[:, :-1]

    return shifted


def generate_dataset(chars: List[str],
                     size: int,
                     noise_prob: float = None,
                     shift_prob: float = 0.5,
                     pad_to: int = 64) -> Dict[str, np.ndarray]:
    """
    Generate a dataset with multiple variations of requested characters.

    Creates 'size' total samples, distributed evenly across characters.
    Each sample is an augmented version (shifted + noisy) of a base character.

    Args:
        chars: List of characters to include (e.g., ['A', 'B', 'C'])
        size: Total number of samples to generate
        noise_prob: Bit flip probability (random 0.05-0.15 if None)
        shift_prob: Probability of shifting (default 0.5)
        pad_to: Pad inputs to this many channels (default 64 for standard brain)

    Returns:
        Dictionary with:
        - 'inputs': (size, pad_to) array of augmented bitmaps
        - 'labels': (size,) array of integer labels (0, 1, 2, ...)
        - 'char_map': Dictionary mapping indices to characters

    Example:
        >>> # Generate 1000 samples of A, B, C
        >>> data = generate_dataset(['A', 'B', 'C'], 1000)
        >>> data['inputs'].shape
        (1000, 64)
        >>> data['labels'].shape
        (1000,)
        >>> data['char_map']
        {0: 'A', 1: 'B', 2: 'C'}
    """
    if not chars:
        raise ValueError("Must provide at least one character")

    # Normalize characters to uppercase
    chars = [c.upper() for c in chars]

    # Validate all characters
    for char in chars:
        if char not in CHARACTER_BITMAPS:
            raise ValueError(f"Character '{char}' not supported")

    # Create character to index mapping
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    # Calculate samples per character
    samples_per_char = size // len(chars)
    remainder = size % len(chars)

    inputs = []
    labels = []

    # Generate samples for each character
    for char_idx, char in enumerate(chars):
        # Get base bitmap
        base_bitmap = get_character_bitmap(char)

        # Number of samples for this character
        num_samples = samples_per_char
        if char_idx < remainder:
            num_samples += 1

        # Generate augmented variations
        for _ in range(num_samples):
            augmented = augment_data(base_bitmap, noise_prob, shift_prob)

            # Pad to standard brain size (64 channels)
            if pad_to > 64:
                padded = np.zeros(pad_to)
                padded[:64] = augmented
            else:
                padded = augmented

            inputs.append(padded)
            labels.append(char_idx)

    # Convert to numpy arrays
    inputs = np.array(inputs, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # Shuffle the dataset
    shuffle_idx = np.random.permutation(size)
    inputs = inputs[shuffle_idx]
    labels = labels[shuffle_idx]

    return {
        'inputs': inputs,
        'labels': labels,
        'char_map': idx_to_char
    }


# ============================================================================
# Convenience Functions
# ============================================================================

def generate_alphabet_dataset(size: int = 1000, **kwargs) -> Dict[str, np.ndarray]:
    """
    Generate dataset with all 26 letters (A-Z).

    Args:
        size: Total number of samples
        **kwargs: Additional arguments for generate_dataset

    Returns:
        Dataset dictionary
    """
    alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    return generate_dataset(alphabet, size, **kwargs)


def generate_digits_dataset(size: int = 1000, **kwargs) -> Dict[str, np.ndarray]:
    """
    Generate dataset with all 10 digits (0-9).

    Args:
        size: Total number of samples
        **kwargs: Additional arguments for generate_dataset

    Returns:
        Dataset dictionary
    """
    digits = list('0123456789')
    return generate_dataset(digits, size, **kwargs)


def generate_alphanumeric_dataset(size: int = 1000, **kwargs) -> Dict[str, np.ndarray]:
    """
    Generate dataset with all 36 alphanumeric characters (A-Z, 0-9).

    Args:
        size: Total number of samples
        **kwargs: Additional arguments for generate_dataset

    Returns:
        Dataset dictionary
    """
    chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    return generate_dataset(chars, size, **kwargs)


def visualize_character(char: str, augmented: bool = False) -> None:
    """
    Print ASCII art visualization of a character bitmap.

    Args:
        char: Character to visualize
        augmented: If True, show augmented version
    """
    bitmap = get_character_bitmap(char)

    if augmented:
        bitmap = augment_data(bitmap)

    grid = bitmap.reshape(8, 8)

    print(f"\nCharacter: {char}")
    print("=" * 20)
    for row in grid:
        line = ''.join(['██' if pixel > 0.5 else '  ' for pixel in row])
        print(line)
    print("=" * 20)


# ============================================================================
# Demo and Testing
# ============================================================================

if __name__ == "__main__":
    # Demo: Generate and visualize characters
    print("\n" + "="*70)
    print("DATA FACTORY DEMO - Procedural Character Generation")
    print("="*70)

    # Show some perfect characters
    print("\n1. Perfect Character Bitmaps:")
    for char in ['A', 'B', 'C', '1', '2', '3']:
        visualize_character(char, augmented=False)

    # Show augmented versions
    print("\n2. Augmented (Noisy) Versions:")
    for char in ['A', 'B', 'C']:
        visualize_character(char, augmented=True)

    # Generate datasets
    print("\n3. Dataset Generation:")
    print("-" * 70)

    # Small alphabet dataset
    data_abc = generate_dataset(['A', 'B', 'C'], size=300)
    print(f"ABC Dataset:")
    print(f"  Inputs shape: {data_abc['inputs'].shape}")
    print(f"  Labels shape: {data_abc['labels'].shape}")
    print(f"  Character map: {data_abc['char_map']}")
    print(f"  Label distribution: {np.bincount(data_abc['labels'])}")

    # Digits dataset
    data_digits = generate_digits_dataset(size=500)
    print(f"\nDigits Dataset:")
    print(f"  Inputs shape: {data_digits['inputs'].shape}")
    print(f"  Labels shape: {data_digits['labels'].shape}")
    print(f"  Number of classes: {len(data_digits['char_map'])}")

    # Full alphabet
    data_alphabet = generate_alphabet_dataset(size=1000)
    print(f"\nFull Alphabet Dataset:")
    print(f"  Inputs shape: {data_alphabet['inputs'].shape}")
    print(f"  Number of classes: {len(data_alphabet['char_map'])}")

    # Alphanumeric
    data_alphanum = generate_alphanumeric_dataset(size=1000)
    print(f"\nAlphanumeric Dataset:")
    print(f"  Inputs shape: {data_alphanum['inputs'].shape}")
    print(f"  Number of classes: {len(data_alphanum['char_map'])}")

    print("\n" + "="*70)
    print("Features:")
    print("  ✓ 36 characters (A-Z, 0-9) hardcoded as 8x8 bitmaps")
    print("  ✓ Random shifts (up/down/left/right by 1 pixel)")
    print("  ✓ Random noise (5-15% bit flips)")
    print("  ✓ Infinite variations - no two samples exactly the same")
    print("  ✓ Automatic padding to 64 channels (standard brain size)")
    print("  ✓ Balanced class distribution")
    print("="*70)

