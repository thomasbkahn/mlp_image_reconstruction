- try fourier encoding (basic form)
x extrapolate beyond bounds of image (outside of 0, 1)
x "super resolution" - zoom in on region and sample (e.g. 0.1 to 0.2)
x make train script, save checkpoints
- make script to generate reconstruction from checkpoint
- dataset save h and w
- make apply encoding function (called in dataset)
- try limited model (few units, small depth)
  - 4 x 16 is bad, try much more depth
  - try 4 x 64
- try dropout
- try other feature encoding (non-periodic)
- try other color space targets
- try 2x scale factor (on GPU)
- do Goya, Saturn
