"""Tests for pipeline.vr_metadata — XML generation and stereo mode handling."""

import pytest

from pipeline.vr_metadata import SPHERICAL_XML_TEMPLATE, VRMetadataEmbedder


class TestSphericalXml:
    """Test Spherical Video V2 XML template generation."""

    def test_sbs_mode_contains_side_by_side(self):
        embedder = VRMetadataEmbedder(stereo_mode="sbs")
        xml = embedder._spherical_xml(3840, 1920)
        assert "side-by-side" in xml

    def test_tb_mode_contains_top_bottom(self):
        embedder = VRMetadataEmbedder(stereo_mode="tb")
        xml = embedder._spherical_xml(3840, 1920)
        assert "top-bottom" in xml

    def test_tb_mode_does_not_contain_side_by_side(self):
        """TB mode must NOT produce side-by-side in XML."""
        embedder = VRMetadataEmbedder(stereo_mode="tb")
        xml = embedder._spherical_xml(3840, 1920)
        assert "side-by-side" not in xml

    def test_xml_contains_projection_type(self):
        embedder = VRMetadataEmbedder()
        xml = embedder._spherical_xml(3840, 1920)
        assert "equirectangular" in xml

    def test_xml_contains_dimensions(self):
        embedder = VRMetadataEmbedder()
        xml = embedder._spherical_xml(7680, 1920)
        assert "7680" in xml
        assert "1920" in xml

    def test_xml_is_valid_xml(self):
        embedder = VRMetadataEmbedder()
        xml = embedder._spherical_xml(3840, 1920)
        assert xml.startswith("<?xml")
        assert "</rdf:SphericalVideo>" in xml

    def test_xml_contains_spherical_true(self):
        embedder = VRMetadataEmbedder()
        xml = embedder._spherical_xml(3840, 1920)
        assert "<GSpherical:Spherical>true</GSpherical:Spherical>" in xml

    def test_xml_contains_stitched_true(self):
        embedder = VRMetadataEmbedder()
        xml = embedder._spherical_xml(3840, 1920)
        assert "<GSpherical:Stitched>true</GSpherical:Stitched>" in xml

    def test_template_has_stereo_mode_placeholder(self):
        """The XML template must use {stereo_mode} placeholder."""
        assert "{stereo_mode}" in SPHERICAL_XML_TEMPLATE

    def test_unknown_stereo_mode_defaults_to_side_by_side(self):
        embedder = VRMetadataEmbedder(stereo_mode="unknown")
        xml = embedder._spherical_xml(3840, 1920)
        assert "side-by-side" in xml


class TestVRMetadataEmbedder:
    """Test VRMetadataEmbedder class initialization."""

    def test_default_codec(self):
        embedder = VRMetadataEmbedder()
        assert embedder.codec == "h264"

    def test_h265_codec_name(self):
        embedder = VRMetadataEmbedder(codec="h265")
        assert embedder._codec_name() == "libx265"

    def test_h264_codec_name(self):
        embedder = VRMetadataEmbedder(codec="h264")
        assert embedder._codec_name() == "libx264"

    def test_default_stereo_mode(self):
        embedder = VRMetadataEmbedder()
        assert embedder.stereo_mode == "sbs"

    def test_custom_stereo_mode(self):
        embedder = VRMetadataEmbedder(stereo_mode="tb")
        assert embedder.stereo_mode == "tb"

    def test_embed_raises_on_empty_frames(self):
        embedder = VRMetadataEmbedder()
        with pytest.raises(ValueError, match="No frames"):
            embedder.embed_single_frame_batch([], "/tmp/test.mp4")
