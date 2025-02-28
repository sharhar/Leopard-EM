import torch

from leopard_em.pydantic_models.correlation_filters import (
    ArbitraryCurveFilterConfig,
    BandpassFilterConfig,
    PhaseRandomizationFilterConfig,
    PreprocessingFilters,
    WhiteningFilterConfig,
)


def test_whitening_filter_config_default():
    config = WhiteningFilterConfig()
    assert config.enabled is True
    assert config.num_freq_bins is None
    assert config.max_freq == 0.5
    assert config.do_power_spectrum is True


def test_whitening_filter_config_calculate():
    config = WhiteningFilterConfig()
    ref_img_rfft = torch.randn(10, 10, dtype=torch.float32)
    filter_tensor = config.calculate_whitening_filter(ref_img_rfft)
    assert filter_tensor.shape == ref_img_rfft.shape


def test_phase_randomization_filter_config_default():
    config = PhaseRandomizationFilterConfig()
    assert config.enabled is False
    assert config.cuton is None


def test_phase_randomization_filter_config_calculate():
    config = PhaseRandomizationFilterConfig(enabled=True, cuton=0.5)
    ref_img_rfft = torch.randn(10, 10, dtype=torch.float32)
    filter_tensor = config.calculate_phase_randomization_filter(ref_img_rfft)
    assert filter_tensor.shape == ref_img_rfft.shape


def test_bandpass_filter_config_default():
    config = BandpassFilterConfig()
    assert config.enabled is False
    assert config.low_freq_cutoff is None
    assert config.high_freq_cutoff is None
    assert config.falloff is None


def test_bandpass_filter_config_calculate():
    config = BandpassFilterConfig(
        enabled=True, low_freq_cutoff=0.1, high_freq_cutoff=0.5, falloff=0.1
    )
    output_shape = (10, 10)
    filter_tensor = config.calculate_bandpass_filter(output_shape)
    assert filter_tensor.shape == output_shape


def test_arbitrary_curve_filter_config_default():
    config = ArbitraryCurveFilterConfig()
    assert config.enabled is False
    assert config.frequencies is None
    assert config.amplitudes is None


def test_arbitrary_curve_filter_config_calculate():
    config = ArbitraryCurveFilterConfig(
        enabled=True, frequencies=[0.1, 0.5], amplitudes=[1.0, 0.5]
    )
    output_shape = (10, 10)
    filter_tensor = config.calculate_arbitrary_curve_filter(output_shape)
    assert filter_tensor.shape == output_shape


def test_preprocessing_filters_default():
    config = PreprocessingFilters()
    assert isinstance(config.whitening_filter, WhiteningFilterConfig)
    assert isinstance(config.bandpass_filter, BandpassFilterConfig)
    assert isinstance(config.phase_randomization_filter, PhaseRandomizationFilterConfig)
    assert isinstance(config.arbitrary_curve_filter, ArbitraryCurveFilterConfig)


def test_preprocessing_filters_combined_filter():
    config = PreprocessingFilters()
    ref_img_rfft = torch.randn(10, 10, dtype=torch.float32)
    output_shape = (10, 10)
    combined_filter = config.get_combined_filter(ref_img_rfft, output_shape)
    assert combined_filter.shape == output_shape
