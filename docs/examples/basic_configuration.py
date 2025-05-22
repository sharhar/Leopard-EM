# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: leo-em-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Configuring Pydantic models in Python
#
# This example notebook outlines the steps necessary to generate, save, and load configurations for the `match_template` program through Python object and `yaml` files.
# Here, we focus on *how* to create and modify these configurations rather than the underlying code for parsing these configurations and running the program.
#
# **Rationale for using YAML configurations**
#
# While the `Leopard-EM` package provides an object-oriented Python API for extending template matching into more complex workflows, it is useful to have a human-readable, easily editable, and shareable configuration file because:
# 1. It increases reproducibility by keeping a record of exact parameters used for a particular run,
# 2. It can be quickly modified during development, debugging, and testing without changing underlying code, and
# 3. It can be replicated across large datasets (e.g. multiple images with similar configurations) for execution on distributed clusters.
#
# We find that storing configurations in a structured file format strikes a good balance between user-friendliness and programmatic control.

# %% [markdown]
# ## Importing Necessary Classes and Functions
#
# We utilize [Pydantic](https://docs.pydantic.dev/latest/) to create Python objects that parse, validate, and serialize configurations.
# These objects (called Pydantic models) are laid out in a hierarchial structure with a single root "manager" model.
# Below we import all the configuration classes (along with other libraries) we will detail usage of in this notebook.

# %%
"""Leopard-EM config objects in Python."""

from pprint import pprint

from leopard_em.pydantic_models.config import (
    ArbitraryCurveFilterConfig,
    BandpassFilterConfig,
    ComputationalConfig,
    DefocusSearchConfig,
    OrientationSearchConfig,
    PhaseRandomizationFilterConfig,
    PreprocessingFilters,
    WhiteningFilterConfig,
)
from leopard_em.pydantic_models.data_structures import OpticsGroup
from leopard_em.pydantic_models.managers import MatchTemplateManager
from leopard_em.pydantic_models.results import MatchTemplateResult

# %% [markdown]
# ## The **OpticsGroup** Model
#
# The `OpticsGroup` model is a container for microscope imaging parameters necessary for calculating filters (e.g. contrast transfer functions).
# We follow the fields that are defined in [RELION's](https://relion.readthedocs.io/en/latest/) optics group .star file, and the class has the following attributes:
# - `label`: A unique label for the optics group, usually contains some form of the micrograph name but can be any string.
# - `pixel_size`: Float value representing the pixel size of the image, in Angstroms.
# - `voltage`: The voltage of the microscope, in kV.
# - `spherical_aberration`: The spherical aberration of the microscope, in mm, with the default value of 2.7 mm.
# - `amplitude_contrast_ratio`: The amplitude contrast ratio (unitless) with the default value of 0.07.
# - `phase_shift`: Additional phase shift to apply across the CTF, in degrees, with the default value of 0.0.
# - `defocus_u`: Defocus of the micrograph along the major axis, in Angstroms.
# - `defocus_v`: Defocus of the micrograph along the minor axis, in Angstroms.
# - `astigmatism_angle`: Angle of the defocus astigmatism (relative to the x-axis), in degrees. The default value is 0.0.
# - `ctf_B_factor`: An additional b-factor to apply to the CTF, in Angstroms^2. The default value is 0.0.
#
# There are other unused fields in the class that are not detailed here.
# See the Pydantic model API documentation for more information.

# %% [markdown]
# ### Creating an instance of the **OpticsGroup** model
#
# Below, we create an instance of the `OpticsGroup` model with some made-up but nevertheless realistic values.

# %%
my_optics_group = OpticsGroup(
    label="my_optics_group",
    pixel_size=1.06,
    voltage=300.0,
    spherical_aberration=2.7,  # default value
    amplitude_contrast_ratio=0.07,  # default value
    phase_shift=0.0,  # default value
    defocus_u=5200.0,
    defocus_v=4950.0,
    astigmatism_angle=25.0,
    ctf_B_factor=60.0,
)

# %% [markdown]
# The Python variable `my_optics_group` is now an instance of the `OpticsGroup` model.
# Note that the model does do validation under-the-hood to ensure necessary fields are present and valid.
# Any invalid fields will raise a `ValidationError` when the model is created.
# Uncomment the following code block to see this in action.

# %%
# bad_optics_group = OpticsGroup(
#     label="bad_optics_group",
#     pixel_size=-1.0,  # <--- Must be positive
#     voltage=300.0,
#     phase_shift=0.0,  # default value
#     defocus_u=5200.0,
#     defocus_v=4950.0,
#     astigmatism_angle=25.0,
# )

# %% [markdown]
# ### Serializing an instance of the **OpticsGroup** model
#
# Pydantic has built-in functionality, namely the `model_dump()`, for generating a dictionary of key, value pairs from the model attributes and their values.
# Below, we create a dictionary from the `my_optics_group` instance and print it out.
# Note that extra, unused fields are still included in the dictionary.

# %%
optics_dict = my_optics_group.model_dump()
pprint(optics_dict)

# %% [markdown]
# ### Exporting configurations to a YAML file
#
# [YAML](https://yaml.org) files are nothing more than a bunch of key-value pairs in a human-readable format.
# Like [JSON](https://www.json.org), YAML has parser functions/libraries in most programming languages increasing their interoperability.
# We adopt the `.yaml` format (and actually the `.json` format too, but not detailed here) for our configuration files rather than less-common formats specific to a sub-field or program.
#
# The `OpticsGroup` model (and all the other Pydanic models discussed here) have a `to_yaml()` method that writes the model to a YAML file.
# Below, we first specify a path and then call the `to_yaml()` method on the `my_optics_group` instance to write the model to a file.

# %%
yaml_filepath = "./optics_group_example.yaml"
my_optics_group.to_yaml(yaml_filepath)

# %% [markdown]
# A new file called `optics_group_example.yaml` should now exist in the current working directory with the following contents:
#
# ----
#
# ```yaml
# amplitude_contrast_ratio: 0.07
# beam_tilt_x: null
# beam_tilt_y: null
# chromatic_aberration: 0.0
# ctf_B_factor: 60.0
# astigmatism_angle: 25.0
# defocus_u: 5200.0
# defocus_v: 4950.0
# even_zernike: null
# label: my_optics_group
# mtf_reference: null
# mtf_values: null
# odd_zernike: null
# phase_shift: 0.0
# pixel_size: 1.06
# spherical_aberration: 2.7
# voltage: 300.0
# zernike_moments: null
# ```
#
# ----

# %% [markdown]
# ### Importing configurations from a YAML file
#
# Each model also has the `from_yaml()` method which can be to instantiate the class from contents in a `.yaml` file.
# Below, we are creating a new instance of the `OpticsGroup` class from the `optics_group.yaml` file.

# %%
new_optics_group = OpticsGroup.from_yaml(yaml_filepath)

# Check if the two OpticsGroup objects are equal
if new_optics_group == my_optics_group:
    print("OpticsGroup objects are equal.")
else:
    print("The two OpticsGroup are not equal!!!")

# %% [markdown]
# Now that we've covered the basics of creating, serializing, and deserializing the `OpticsGroup` model, we can move onto the next models without covering the (de)serialization and import/export steps in detail.

# %% [markdown]
# ## The **OrientationSearchConfig** Model
#
# Two-dimensional template matching necessitates covering SO(3) orientation space to find the "best" orientation match for a particle.
# How points are sampled during the search process is handled by the `OrientationSearchConfig` model.
# This model effectively acts as an interface with the [torch-so3](https://github.com/teamtomo/torch-so3) package, which provides the underlying functionality for generating uniform grids on SO(3).
#
# The class has the following attributes:
#  - `psi_step`: The psi step size (in units of degrees) with a default value of 1.5 degrees.
#  - `theta_step`: The theta step size (in units of degrees) with a default value of 2.5 degrees.
#  - `phi_min`: The minimum value for the $\phi$ Euler angle (in degrees) with a default value of 0.0.
#  - `phi_max`: The maximum value for the $\phi$ Euler angle (in degrees) with a default value of 360.0.
#  - `theta_min`: The minimum value for the $\theta$ Euler angle (in degrees) with a default value of 0.0.
#  - `theta_max`: The maximum value for the $\theta$ Euler angle (in degrees) with a default value of 180.0.
#  - `psi_min`: The minimum value for the $\psi$ Euler angle (in degrees) with a default value of 0.0.
#  - `psi_max`: The maximum value for the $\psi$ Euler angle (in degrees) with a default value of 360.0.
#  - `base_grid_method`: The method used to generate the base S2 grid. Allowed values are `"uniform"` and `"healpix"`. The default value is `"uniform"`.
#  - `symmetry`: Specify the template symmetry to automatically restrict the angular search space. Default `"C1"`. Note, this is ignored if manual angular ranges are given.
#
# Note that the default min/max values set the search space to cover SO(3) for a particle with `"C1"` symmetry.
#

# %% [markdown]
# Below, we create a new instance of the `OrientationSearchConfig` model with only the `psi_step` and `theta_step` attributes set to non-default values.

# %%
orientation_search_config = OrientationSearchConfig(
    psi_step=4.0,
    theta_step=4.0,
)

# print the model dictionary
orientation_search_config.model_dump()

# %% [markdown]
# ## The **DefocusSearchConfig** Model
#
# Two-dimensional template matching is also sensitive to the relative defocus of a particle allowing the estimation of the Z-height in a sample.
# The `DefocusSearchConfig` model handles which defocus planes are searched over (relative to the defocus parameters defined in the `OpticsGroup` model).
#
# The model has the following attributes:
#  - `enabled`: A boolean value indicating whether defocus search is enabled. The default value is `True`. If `False`, then only the defocus value defined in the `OpticsGroup` model is used.
#  - `defocus_min`: The minimum relative defocus value (in Angstroms) to search.
#  - `defocus_max`: The maximum relative defocus value (in Angstroms) to search.
#  - `defocus_step`: The increment between searched defocus planes (in Angstroms).
#
# These parameters will generate a set of relative defocus planes searched over according to
#     $$[\text{f}_\text{min}, \text{f}_\text{min} + \Delta\text{f}, + \text{f}_\text{min} + 2\Delta\text{f}, \dots, \text{f}_\text{max}]$$
# which is effectively the following range object in Python:
# ```python
# range(defocus_min, defocus_max + defocus_step, defocus_step)
# ```

# %%
# Searches defocus between -600 and 600 with a step of 200 Angstroms
defocus_search_config = DefocusSearchConfig(
    enabled=True, defocus_min=-600, defocus_max=600, defocus_step=200
)

# %% [markdown]
# ### The `DefocusSearchConfig.defocus_values` property
#
# Once a `DefocusSearchConfig` model is instantiated, there is the helpful `defocus_values` property that returns a list of relative defocus values to search over.

# %%
defocus_search_config.defocus_values

# %% [markdown]
# ## Fourier filters in the **PreprocessingFilters** Model
#
# Template matching necessitates the use of Fourier filters to preprocess the input image (e.g. spectral whitening).
# The `PreprocessingFilters` model handles the configuration of the following filter types:
#   - Spectral whitening under the `whitening_filter` attribute
#   - Bandpass filtering, with the option for smooth transitions, under the `bandpass_filter` attribute.
#   - Phase randomization above a certain frequency using the `phase_randomization_filter` attribute.
#   - Options for a user-defined arbitrary curve filter under the `arbitrary_curve_filter` attribute.
#
# Together, all these filter types allow fine control over how an input image is preprocessed before template matching.
# Each filter type is itself a Pydantic model with its own set of attributes.

# %% [markdown]
# ### **WhiteningFilterConfig**
#
# The `WhiteningFilterConfig` model handles the configuration of the spectral whitening filter.
# When applied the image, the power spectral density should become flat and the noise distribution is white (i.e. uncorrelated).
#
# The whitening filter is *enabled* by default and has the following attributes:
#   - `enabled`: A boolean value indicating whether the whitening filter is enabled.
#   - `num_freq_bins`: An optional integer specifying the number of frequency bins used when calculating the power spectral density. This parameter is automatically calculated based on the input image size if not provided.
#   - `max_freq`: An optional float specifying the maximum spatial frequency (in terms of Nyquist) to use when calculating the whitening filter. Frequencies above this value are set to `1.0`, that is, unscaled. The default value is `0.5` which corresponds to the Nyquist frequency.
#   - `do_power_spectrum`: Boolean indicating weather the whitening filter should be calculated over the power spectrum *or* amplitude spectrum. The default value is `True` and the power spectrum is used.
#
# Below, we create a default instance of the `WhiteningFilterConfig` model.

# %%
whitening_filter_config = WhiteningFilterConfig()

# %% [markdown]
# ### **BandpassFilterConfig**
#
# The `BandpassFilterConfig` model handles the configuration of the bandpass filter.
#
# The bandpass filter is *disabled* by default and has the following attributes:
#   - `enabled`: A boolean value indicating whether the bandpass filter is enabled.
#   - `low_freq_cutoff`: The low-pass cutoff frequency (in terms of Nyquist) for the bandpass filter.
#   - `high_freq_cutoff`: The high-pass cutoff frequency (in terms of Nyquist) for the bandpass filter.
#   - `falloff`: The falloff factor (using a cosine function) for the bandpass filter. A value of `0.0` (default) corresponds to a hard cutoff with values in the range `(0.0, 0.1)` providing a smooth, but distinct, transition.
#
# When disabled, the bandpass filter is not applied to the input image.
# Nonetheless, we create a default instance of the `BandpassFilterConfig` model below.

# %%
bandpass_filter_config = BandpassFilterConfig()

# %% [markdown]
# ### **PhaseRandomizationFilterConfig**
#
# The `PhaseRandomizationFilterConfig` model hold parameters defining a phase randomization filter.
# This filter keeps the amplitudes of Fourier components above a certain frequency the same, but randomizes their phases. This is useful for testing the robustness of template matching algorithms to noise.
#
# The model is *disabled* by default has the following attributes:
#   - `enabled`: A boolean value indicating whether the phase randomization filter is enabled.
#   - `cuton`: The cuton frequency (in terms of Nyquist) for the phase randomization filter. Frequencies above this value are randomized.
#
# Below, we create a default instance of the `PhaseRandomizationFilterConfig` model.

# %%
phase_randomization_filter = PhaseRandomizationFilterConfig()

# %% [markdown]
# ### **ArbitraryCurveFilterConfig**
#
# We also provide a model for defining an arbitrary curve filter for preprocessing.
# This filter takes a set of spatial frequency values (in terms of Nyquist) and filter amplitudes at those frequencies to create a custom filter.
# Utilizing this filter allows for fine-grained control over how spatial frequencies should be weighted within the template matching package itself.
#
# The model is *disabled* by default has the following attributes:
#  - `enabled`: A boolean value indicating whether the arbitrary curve filter is enabled.
#  - `frequencies`: 1-dimensional list of floats representing the spatial frequencies (in terms of Nyquist). The list must be sorted in ascending order.
#  - `amplitudes`: 1-dimensional list of floats representing the filter amplitudes at the corresponding frequencies. The list must be the same length as `frequencies`.
#
# Below, we create a default instance of the `ArbitraryCurveFilterConfig` mode; it is disabled and has no frequencies or amplitudes set.

# %%
arbitrary_curve_filter = ArbitraryCurveFilterConfig()

# %% [markdown]
# ### Putting the filters together in the **PreprocessingFilters** Model
#
# We now construct the `PreprocessingFilters` model with the instances of the four filter models we created above.

# %%
preprocessing_filters = PreprocessingFilters(
    whitening_filter=whitening_filter_config,
    bandpass_filter=bandpass_filter_config,
    phase_randomization_filter=phase_randomization_filter,
    arbitrary_curve_filter=arbitrary_curve_filter,
)

# %% [markdown]
# ## **ComputationalConfig**
#
# The `ComputationalConfig` model currently only handles the GPU ids to use for template matching.
# The model has the following attributes:
#  - `gpu_ids`: A list of integers representing the GPU ids to use for template matching. The default value is `[0]` which corresponds to the first GPU.
#
# Below, we create a new instance of the `ComputationalConfig` model with the default GPU id list.

# %%
comp_config = ComputationalConfig()
comp_config

# %% [markdown]
# ## Specifying result output with the **MatchTemplateResult** Model
#
# We almost have a complete set of configurations for the `match-template` program, but we still need to specify where to save results after the program completes.
# The `MatchTemplateResult` model handles this by specifying output file paths.
# The model also has handy class methods for analyzing results and picking particles, but this is discussed elsewhere in the documentation.
#
# ### User-definable attributes
#
# The model has the following user-definable attributes:
#   - `allow_file_overwrite`: A boolean value indicating whether the program should be allowed to overwrite existing files. The default value is `False` and will raise an error if a file already exists.
#   - `mip_path`: The path to save the maximum intensity projection (MIP) image.
#   - `scaled_mip_path`: The path to save the scaled MIP (a.k.a z-score or SNR) image.
#   - `correlation_average_path`: The path to save the average correlation value per pixel.
#   - `correlation_variance_path`: The path to save the variance of the correlation value per pixel.
#   - `orientation_psi_path`: The path to save the best $\psi$ Euler angle map.
#   - `orientation_theta_path`: The path to save the best $\theta$ Euler angle map.
#   - `orientation_phi_path`: The path to save the best $\phi$ Euler angle map.
#   - `relative_defocus_path`: The path to save the best relative defocus map.
#
# ### Attributes updated after template matching
#
# There are additional attributes in the model which automatically get updated after template matching is complete:
#   - `total_projections`: The total number of projections \(\text{orientations} \times \text{defocus planes}\) searched over.
#   - `total_orientations`: The total number of orientations searched over.
#   - `total_defocus`: The total number of defocus planes searched over.
#
# ### Creating an instance of the **MatchTemplateResult** model
#
# Below, we specify the necessary output paths for the `MatchTemplateResult` model.
# Note that this configuration will output the images into thee current working directory.
# You will need to update these paths to whatever is appropriate for your system.

# %%
match_template_result = MatchTemplateResult(
    allow_file_overwrite=True,
    mip_path="./output_mip.mrc",
    scaled_mip_path="./output_scaled_mip.mrc",
    correlation_average_path="./output_correlation_average.mrc",
    correlation_variance_path="./output_correlation_variance.mrc",
    orientation_psi_path="./output_orientation_psi.mrc",
    orientation_theta_path="./output_orientation_theta.mrc",
    orientation_phi_path="./output_orientation_phi.mrc",
    relative_defocus_path="./output_relative_defocus.mrc",
)

# %% [markdown]
# ## Root **MatchTemplateConfig** Model
#
# Finally, we have all the components which go into the root `MatchTemplateConfig` model.
# This model is the top-level configuration object that contains all the other models as attributes along with `micrograph_path` and `template_volume_path` which point to the input micrograph and simulated reference template volume, respectfully.
#
# Below, we create our instance of the `MatchTemplateConfig` model.
# Note that you will need to supply the paths to the micrograph and template volume on your system; dummy paths are provided here so the code runs without error.
#

# %%
match_template_manager = MatchTemplateManager(
    micrograph_path="./dummy_micrograph.mrc",
    template_volume_path="./dummy_template_volume.mrc",
    optics_group=my_optics_group,
    defocus_search_config=defocus_search_config,
    orientation_search_config=orientation_search_config,
    preprocessing_filters=preprocessing_filters,
    match_template_result=match_template_result,
    computational_config=comp_config,
    preload_mrc_files=False,  # Don't try to read the MRC upon initialization
)

# %% [markdown]
# ### Serializing the **MatchTemplateConfig** model
#
# Like discussed before, we can serialize and read the `MatchTemplateConfig` model to/from a YAML file.
# Below, we write the model to a file called `match_template_example.yaml`.

# %%
match_template_manager.to_yaml("./match_template_manager_example.yaml")

# %% [markdown]
# ### Importing the **MatchTemplateConfig** model from a YAML file
#
# Now, we re-import the configuration into a new model and check that they are the same.
#

# %%
new_match_template_manager = MatchTemplateManager.from_yaml(
    "./match_template_manager_example.yaml"
)

if new_match_template_manager == match_template_manager:
    print("MatchTemplateManager objects are equal.")
else:
    print("The two MatchTemplateManager are not equal!!!")

# %% [markdown]
# ## Conclusion
#
# We have now covered the creation, serialization, and deserialization of all the configuration models necessary for the `match-template` program.
# This script will create the `match_template_example.yaml` file in the current working directory whose file contents should match what is listed below.
# Modifying this file and using it as input to the `match-template` program will allow you to run the program with the specified configurations.
# Note that a default YAML configuration can also be found in the GitHub page.
#
# ----
#
# ```yaml
# computational_config:
#   gpu_ids:
#   - 0
#   num_cpus: 1
# defocus_search_config:
#   defocus_max: 600.0
#   defocus_min: -600.0
#   defocus_step: 200.0
#   enabled: true
# match_template_result:
#   allow_file_overwrite: true
#   correlation_average_path: ./output_correlation_average.mrc
#   correlation_variance_path: ./output_correlation_variance.mrc
#   mip_path: ./output_mip.mrc
#   orientation_phi_path: ./output_orientation_phi.mrc
#   orientation_psi_path: ./output_orientation_psi.mrc
#   orientation_theta_path: ./output_orientation_theta.mrc
#   relative_defocus_path: ./output_relative_defocus.mrc
#   scaled_mip_path: ./output_scaled_mip.mrc
#   total_defocus: 0
#   total_orientations: 0
#   total_projections: 0
# micrograph_path: ./dummy_micrograph.mrc
# optics_group:
#   amplitude_contrast_ratio: 0.07
#   beam_tilt_x: null
#   beam_tilt_y: null
#   chromatic_aberration: 0.0
#   ctf_B_factor: 60.0
#   astigmatism_angle: 25.0
#   defocus_u: 5200.0
#   defocus_v: 4950.0
#   even_zernike: null
#   label: my_optics_group
#   mtf_reference: null
#   mtf_values: null
#   odd_zernike: null
#   phase_shift: 0.0
#   pixel_size: 1.06
#   spherical_aberration: 2.7
#   voltage: 300.0
#   zernike_moments: null
# orientation_search_config:
#   base_grid_method: uniform
#   psi_step: 4.0
#   theta_step: 4.0
#   phi_max: 360.0
#   phi_min: 0.0
#   psi_max: 360.0
#   psi_min: 0.0
#   theta_max: 180.0
#   theta_min: 0.0
# preprocessing_filters:
#   arbitrary_curve_filter:
#     amplitudes: null
#     enabled: false
#     frequencies: null
#   bandpass_filter:
#     enabled: false
#     falloff: null
#     high_freq_cutoff: null
#     low_freq_cutoff: null
#   phase_randomization_filter:
#     cuton: null
#     enabled: false
#   whitening_filter:
#     do_power_spectrum: true
#     enabled: true
#     max_freq: 0.5
#     num_freq_bins: null
# template_volume_path: ./dummy_template_volume.mrc
# ```
#
# ----
#
