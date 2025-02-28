import pytest

from leopard_em.pydantic_models.defocus_search import DefocusSearchConfig


def test_default_values():
    config = DefocusSearchConfig(
        defocus_min=-1000.0, defocus_max=1000.0, defocus_step=100.0
    )
    assert config.enabled is True
    assert config.defocus_min == -1000.0
    assert config.defocus_max == 1000.0
    assert config.defocus_step == 100.0


def test_defocus_values():
    config = DefocusSearchConfig(
        defocus_min=-1000.0, defocus_max=1000.0, defocus_step=100.0
    )
    assert config.defocus_values == [
        -1000,
        -900,
        -800,
        -700,
        -600,
        -500,
        -400,
        -300,
        -200,
        -100,
        0,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
    ]


def test_defocus_values_disabled():
    config = DefocusSearchConfig(
        enabled=False, defocus_min=-1000.0, defocus_max=1000.0, defocus_step=100.0
    )
    assert config.defocus_values == [0.0]


def test_defocus_values_no_values():
    with pytest.raises(
        ValueError,
        match="Defocus search parameters result in no values to search over!",
    ):
        df_config = DefocusSearchConfig(
            defocus_min=1000.0, defocus_max=-1000.0, defocus_step=100.0
        )
        _ = df_config.defocus_values


def test_defocus_values_with_step():
    config = DefocusSearchConfig(
        defocus_min=-400.0, defocus_max=400.0, defocus_step=200.0
    )
    assert config.defocus_values == [-400, -200, 0, 200, 400]
