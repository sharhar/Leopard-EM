# TODO list for template matching

Some items to consider both in terms of boosting the sensitivity of 2DTM, improving general accessibility of the method, and increasing the speed/efficiency of the code.

## Essential TODO items

Items that need completed before releasing the codebase to a general audience.

### Codebase
- [x] Test GPU code (fix all bugs) then benchmark speed
- [x] Tidy up the tt2DTM code a lot
  - [x] Put in the pydantic models for input/output
  - [x] Test the program input/outputs
- [ ] Write program for make template results
- [x] Write program for refine template

### Documentation
 - [x] Create example notebook for how to use Pydantic models
 - [x] Write an example python script for running a single match template job
 - [x] Deploy some rudimentary documentation on readthedocs

### Maintainability and Organization
 - [ ] Write basic unit tests for the Pydantic models & associated methods
 - [ ] Write basic unit tests for the pre-processing steps/functions
 - [ ] Add complexity limit within pre-commit hooks (improves readability of functions)
 - [x] Ensure `pyproject.toml` has strict versioning requirements which work across platform(s) -- should we target anything but Linux?

## Increasing efficiency/speed

 Items related to increasing the speed of the template matching process, especially for the whole orientation search.
 - [ ] Write Vulkan and/or CUDA code and compare speed 
- ~~[ ] Test reducing the floating point precision~~ --> PyTorch natively does not support float16/complex32 operations for FFTs. Would need to have non-torch backend to test this.
 - [ ] Binning the input micrograph to a larger pixel size based on the maximum resolution. Will need to benchmark sensitivity.
 - Apply Fourier filters to micrograph only (creating a stack of micrographs to search over) thus reducing the number of FFTs for templates.

## Increasing sensitivity
 - [ ] Compare results with and without padding for Fourier slice extraction, see what padding is minimal.
 - [ ] Some testing with Fourier filters:
  - [ ] Try whitening filter with amplitude vs intensity
  - [ ] Smoothing vs unsmoothed whitening filter
  - [ ] Phase randomization above certain resolution
  - [ ] Hard high-pass filter
  - [ ] Smoothed band-pass filter (down weighting low resolution information, but not entirely removing it)
  - [ ] A more explicit weighting function like gisSPA
- [ ] Other correlation metrics other than the maximum cross-correlation
- [ ] Per particle motion correction 
  - [ ] Tilt series alignment refinement
- [ ] Try signal subtraction of refined targets
- [ ] Put in the p-score metric

### Functionality
- [ ] Try things other than maximal cross correlation
- [ ] Re-dose weight particle movies
