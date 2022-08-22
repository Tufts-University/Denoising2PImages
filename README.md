# Denoising Cervical-Tissue, Two-Photon Images
## Repository Summary
The following code can be used to implement competing models of denoising on 2D images. These particular algorithms were designed for use on two-photon images acquired by a Leica SP8 microscope. Included functions allow for customization of loss functions, trianing, and eval only. All implementation for both the RCAN [[1]](#1) and CARE [[2]](#2) algorithms was done in TensorFlow. Work is currently being done to improve model performance and include novel algorithms such as SRGAN [[3]](#3) which have shown high levels of performance in other studies. For more details on RCAN, CARE, or SRGAN please review the cited papers below. We would like to thank the authors of these papers for inspiring our work and providing codes to implement these models online.



## References
<a id="1">[1]</a>
Chen, J., Sasaki, H., Lai, H. *et al.* Three-dimensional residual channel attention networks denoise and sharpen fluorescence microscopy image volumes. *Nat Methods* **18**, 678-687 (2021). https://doi.org/10.1038/s41592-021-01155-x

<a id="2">[2]</a>
Weigert, M., Schmidt, U., Boothe, T. *et al.* Content-aware image restoration: pushing the limits of fluorescence microscopy. *Nat Methods* **15**, 1090–1097 (2018). https://doi.org/10.1038/s41592-018-0216-7

<a id="3">[3]</a>
Ledig, C., Theis, L., Huszar, F. *et al.* Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. *arXiv* (2016). https://doi.org/10.48550/arxiv.1609.04802

