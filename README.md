# Denoising Cervical-Tissue, Two-Photon Images
## Repository Summary
The following code can be used to implement competing models of denoising on 2D images. These particular algorithms were designed for use on two-photon images acquired by a Leica SP8 microscope. Included functions allow for customization of loss functions, trianing, and eval only. All implementation for both the RCAN [[1]](https://doi.org/10.1038/s41592-021-01155-x) and CARE algorithms was done in TensorFlow. Work is currently being done to improve model performance and include novel algorithms such as SRGAN which have shown high levels of performance in other studies. For more details on RCAN, CARE, or SRGAN please review the cited papers below. We would like to thank the authors of these papers for inspiring our work and providing codes to implement these models online.



## References
<a id="1">[1]</a>
Chen, J., Sasaki, H., Lai, H. *et al.* Three-dimensional residual channel attention networks denoise and sharpen fluorescence microscopy image volumes. *Nat Methods* **18**, 678-687 (2021). https://doi.org/10.1038/s41592-021-01155-x
