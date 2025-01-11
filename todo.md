# tt2DTM todo

Things left to do for tt2DTM

### Essentials Todo

- [ ] Test GPU code (fix all bugs) then benchmark speed
- [ ] Write Vulkan code and compare speed 
- [ ] Tidy up the tt2DTM code a lot
  - [ ] Put in the pydantic models for input
- [ ] Write the make template results
- [ ] Write the refine template

### Speed things

- [ ] Test reducing the floating point precision
- [ ] Bin micrograph to larger pixel sizes based on a max res
- [ ] Apply CTF to the micrograph instead of template
- [ ] Simulate in program and then crop to size of potential. Ensure this is as small as possible at each step
- [ ] Compare results with and without padding, see what padding is minimal.
- [ ] PCA

### Functionality
- [ ] Put in the p-score metric
- [ ] Play around with the filters
  - [ ] Try whitening filter with amplitude vs intensity
  - [ ] Smoothing vs unsmoothed whitening filter
  - [ ] Phase randomization 
  - [ ] High pass filters
  - [ ] A more explicit weighting function like gisSPA
- [ ] Try signal subtraction of refined targets
- [ ] Try things other than maximal cross correlation
- [ ] Per particle motion correction 
  - [ ] Tilt series alignment refinement
- [ ] Re-dose weight particle movies



### Done ✓

- [x] CPU code 