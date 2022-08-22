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
    ```
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
## Training
To train any model, you simply need to specify the run options and call the [main.py](main.py) function:

    python -u main.py train *model_architecture*  "*Model_Name*" cwd=*cwd* nadh_data=*nadh_data.npz* loss=*Loss_fcn* wavelet_function=*Wavelet_transform* loss_alpha=*SSIM contributation*

Available options include:
- `Mode`:
    - *train* - used when training a new model
    - *eval* - used to evaluate with a set of trained weights. Must have `weights_final.hdf5` saved in a folder called *Model_Name*.
- `Model_Name`:
    - Specify the name of the model you are training. Good notion includes information about specific parameters used i.e.: 'NADH_CAREModel_SSIMR2Loss_alpha_p5_wavelet_bior1p1'
    - Here, we include what type of data the model will be trained on (NADH or FAD), which model (CARE or RCAN), which loss function (see below), the alpha level of each weight included, if wavelet trandorm will be used and if so which type.
- `cwd`:
    - List the path where the data (.npz) files are stored for training **(default = ' ')**
- `nadh_data`:
    - List the name of the NADH data file during training for NADH training **(default = ' ')**
- `fad_data`:
    - List the name of the NADH data file during training for FAD training **(default = ' ')**
- `epochs`:
    - Maximum number of epochs desired **(default = 300)**
- `steps_per_epoch`:
    - Number of steps per epoch, this controls the batch size of the model **(default = None for rcan or 100 for care)**
- `input_shape`:
    - Set to the patch size used for your images **(default = [256 256])**
- `initial_learning_rate`:
    - Starting learning rate used for training **(default = 1e-5)**
- `loss`:
    - Multiple options for loss functions exist:
        - *ssim_loss*
        - *ssiml1_loss* **(default)**
        - *ssimr2_loss*
        - *MSSSIM_loss* **(must preinitalize some weights)**
        - *pcc_loss*
        - *ssimpcc_loss*
- `metrics`: 
    - Multiple options for metrics exist:
        - *psnr*
        - *ssim*
        - *pcc* 
    - **Default: ['psnr', 'ssim']**
- `loss_alpha`:
    - Controls how much different loss functions are weighted in the compound loss **(Default = 0.5)**
- `wavelet_function`:
    - If analysis included wavelet transformation of the data, specify which mother wavelet to use
        - chose from `pywt.wavelist()`
    - **Default = ' '**
### Model Specfic Config:
**RCAN config**
- `num_channels`: 
    - Number of channels for expansion **(Default = 32)**
- `num_residual_blocks`:
    - Number of residual blocks in each residual group **(Default = 5)**
- `num_residual_groups`:
    - Number of residual groups **(Default = 5)**
- `channel_reduction`: 
    - Specifies the channel reduction factor which defines the number of filters used (`num_channels`/`channel_reduction` = `num_filters`) **(Default  = 4)**

**CARE UNET config**
- `unet_n_depth`:
    - Number of downsampling steps required before upsampling begins **(Default = 6)**
- `unet_n_first`:
    - Number of channels for expansion **(Default = 32)**
- `unet_kern_size`: 3,
    - Kernel size used during convolution steps **(Default = 3)**

## References
<a id="1">[1]</a>
Chen, J., Sasaki, H., Lai, H. *et al.* Three-dimensional residual channel attention networks denoise and sharpen fluorescence microscopy image volumes. *Nat Methods* **18**, 678-687 (2021). https://doi.org/10.1038/s41592-021-01155-x

<a id="2">[2]</a>
Weigert, M., Schmidt, U., Boothe, T. *et al.* Content-aware image restoration: pushing the limits of fluorescence microscopy. *Nat Methods* **15**, 1090–1097 (2018). https://doi.org/10.1038/s41592-018-0216-7

<a id="3">[3]</a>
Ledig, C., Theis, L., Huszar, F. *et al.* Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. *arXiv* (2016). https://doi.org/10.48550/arxiv.1609.04802

