"""
Tests for macro1/utils/utils.py

Tests the utility functions: encode_image_url, contains_non_ascii, smart_resize.
"""

import pytest
import base64
from PIL import Image

from macro1.utils.utils import (
    encode_image_url,
    contains_non_ascii,
    smart_resize,
)


class TestEncodeImageUrl:
    """Tests for encode_image_url function."""

    def test_encode_basic_image(self, mock_image):
        url = encode_image_url(mock_image)
        assert url.startswith("data:image/png;base64,")
        base64_part = url.replace("data:image/png;base64,", "")
        decoded = base64.b64decode(base64_part)
        assert len(decoded) > 0

    def test_encode_with_resize(self, mock_image):
        url = encode_image_url(mock_image, resize=(100, 100))
        assert url.startswith("data:image/png;base64,")

    def test_encode_small_image(self, mock_image_small):
        url = encode_image_url(mock_image_small)
        assert url.startswith("data:image/png;base64,")

    def test_encode_returns_valid_base64(self):
        img = Image.new('RGB', (50, 50), color='red')
        url = encode_image_url(img)
        base64_part = url.split(",", 1)[1]
        decoded = base64.b64decode(base64_part)
        # Should be a valid PNG
        assert decoded[:4] == b'\x89PNG'


class TestContainsNonAscii:
    """Tests for contains_non_ascii function."""

    def test_ascii_only(self):
        assert contains_non_ascii("Hello World!") is False

    def test_with_non_ascii(self):
        assert contains_non_ascii("你好世界") is True
        assert contains_non_ascii("Héllo") is True
        assert contains_non_ascii("日本語") is True

    def test_mixed(self):
        assert contains_non_ascii("Hello 世界") is True

    def test_empty_string(self):
        assert contains_non_ascii("") is False


class TestSmartResize:
    """Tests for smart_resize function."""

    def test_basic_resize(self):
        height, width = smart_resize(1920, 1080)
        assert height > 0
        assert width > 0
        assert height % 28 == 0
        assert width % 28 == 0

    def test_small_image_resize(self):
        height, width = smart_resize(50, 50, min_pixels=3136)
        assert height * width >= 3136

    def test_large_image_resize(self):
        height, width = smart_resize(10000, 10000, max_pixels=1000000)
        assert height * width <= 1000000

    def test_custom_factor(self):
        height, width = smart_resize(100, 100, factor=14)
        assert height % 14 == 0
        assert width % 14 == 0

    def test_extreme_aspect_ratio_raises(self):
        with pytest.raises(ValueError, match="aspect ratio"):
            smart_resize(10000, 1)

    def test_result_divisible_by_factor(self):
        for h, w in [(640, 480), (1920, 1080), (800, 600)]:
            rh, rw = smart_resize(h, w)
            assert rh % 28 == 0
            assert rw % 28 == 0
