# Denoising Cervical-Tissue, Two-Photon Images

## Repository Summary
The following code can be used to implement competing models of denoising on 2D images. These particular algorithms were designed for use on two-photon images acquired by a Leica SP8 microscope. Included functions allow for customization of loss functions, trianing, and eval only. All implementation for both the RCAN [[1]](https://doi.org/10.1038/s41592-021-01155-x) and CARE [[2]](https://doi.org/10.1038/s41592-018-0216-7) algorithms was done in TensorFlow. Work is currently being done to improve model performance and include novel algorithms such as SRGAN [[3]](https://doi.org/10.48550/arxiv.1609.04802) which have shown high levels of performance in other studies. For more details on RCAN, CARE, or SRGAN please review the cited papers below. We would like to thank the authors of these papers for inspiring our work and providing codes to implement these models online.

## System Requirements
- Code was written on a Windows 10 system but run on a Linux system with a A100 GPU installed
- Python 3.8+
- NVIDIA A100 GPU
- CUDA 11.2 and cuDNN 8.1

**Tested Enviroment**:
1. RCAN Network
    - NVIDIA A100 GPU 40GB 
    - 50GB of RAM
    - Training Time: ~ 4.7 hrs 
    - Evaluation Time: ~ 4 minutes
2. CARE Network
    - NVIDIA A100 GPU 40GB 
    - 50GB of RAM
    - Training Time: ~ 40 minutes 
    - Evaluation Time: ~ 2 minutes

## Dataset
1. All raw data is available for at reasonable request. Contact [Professor Irene Georgakoudi](mailto:irene.georgakoudi@tufts.edu) for access to datasets. 
2. Pretrained model weights are stored in Trained Model Folder
3. Preformatted data is avaiable online under Data folder

## Dependencies Installation
### (Option 1) Install dependencies in base environment

1. Download the [`pip_requirements.txt`](pip_requirements.txt) from the repository
2. In your command prompt run:

    ```posh
    pip install -r pip_requirements.txt
    ```
3. Download the [`conda_requirements.txt`](conda_requirements.txt) from the repository
4. In your command prompt run:

    ```posh
    conda install --file conda_requirements.txt
    ```
### (Option 2) Create a new virtual environment

1. Download the [`pip_requirements.txt`](pip_requirements.txt) from the repository
2. Download the [`conda_requirements.txt`](conda_requirements.txt) from the repository
3. Open command prompt and change folder to where you put the `pip_requirements.txt` and `conda_requirements.txt`
4. Create a new virtual environment:

    ```posh
    python -m venv Denoising
5. Activate the virtual environment

    On Windows:

    ```posh
    .\Denoising\Scripts\activate
    ```

    On macOS and Linux:

    ```bash
    source Denoising/bin/activate
    ```

6. You should see (Denoising) in the command line.

7. In your command prompt run:

    ```posh
    pip install -r pip_requirements.txt
    ```
    ```posh
    conda install --file conda_requirements.txt
    ```
   ```

## References
<a id="1">[1]</a>
Chen, J., Sasaki, H., Lai, H. *et al.* Three-dimensional residual channel attention networks denoise and sharpen fluorescence microscopy image volumes. *Nat Methods* **18**, 678-687 (2021). https://doi.org/10.1038/s41592-021-01155-x

<a id="2">[2]</a>
Weigert, M., Schmidt, U., Boothe, T. *et al.* Content-aware image restoration: pushing the limits of fluorescence microscopy. *Nat Methods* **15**, 1090–1097 (2018). https://doi.org/10.1038/s41592-018-0216-7

<a id="3">[3]</a>
Ledig, C., Theis, L., Huszar, F. *et al.* Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. *arXiv* (2016). https://doi.org/10.48550/arxiv.1609.04802

