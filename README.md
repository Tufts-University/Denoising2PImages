# Denoising Cervical-Tissue, Two-Photon Images

## Repository Summary
The following code can be used to implement competing models of denoising on 2D images. These particular algorithms were designed for use on two-photon images acquired by a Leica SP8 microscope. Included functions allow for customization of loss functions, training, and evalulation only. All implementation for RCAN [[1]](https://doi.org/10.1038/s41592-021-01155-x), CARE [[2]](https://doi.org/10.1038/s41592-018-0216-7) and SRGAN [[3]](https://doi.org/10.48550/arxiv.1609.04802) algorithms was done in TensorFlow. For more details on RCAN, CARE, or SRGAN please review the cited papers below. Wavelet-based denoising was also introduced for improved results. The implementation is referred to as WU-net and followed the discussed implementation from *Aytekin et al.* [[4]](https://doi.org/10.1109/MMSP53017.2021.9733576.). We would like to thank the authors of these papers for inspiring our work and providing codes to implement these models online.

## System Requirements
- Code was written on a Windows 10 system but run on a Linux system with a A100 GPU installed
- Python 3.8+
- NVIDIA A100 GPU
- CUDA 11.2 and cuDNN 8.1

**Tested Enviroment**:
All parameters listed for training and validating on 8844 paired image patches (256 x 256) and evaluated on 4072 paired image patches (256 x 256).
1. RCAN Network
    - NVIDIA A100 GPU 40GB 
    - 22GB of RAM 
    - Training Time: ~ 4.7 hrs 
    - Evaluation Time: ~ 40 minutes
2. CARE Network
    - NVIDIA A100 GPU 40GB 
    - 22GB of RAM
    - Training Time: ~ 40 minutes 
    - Evaluation Time: ~ 40 minutes
3. Resnet Network
    - NVIDIA A100 GPU 40GB 
    - 22GB of RAM
    - Training Time: ~ 2 hours 
    - Evaluation Time: ~ 40 minutes
4. SRGAN Network
    - NVIDIA A100 GPU 40GB 
    - 22GB of RAM
    - Training Time: ~ 24 hours 
    - Evaluation Time: ~ 40 minutes
5. WU-net
    - NVIDIA A100 GPU 40GB 
    - 22GB of RAM
    - Training Time: ~ 2 hours 
    - Evaluation Time: ~ 40 minutes

## Dataset
1. All raw data is available at reasonable request. Contact [Professor Irene Georgakoudi](mailto:irene.georgakoudi@tufts.edu) for access to datasets. 
2. Pretrained model weights are stored in Trained Model Folder (*Coming Soon*)
3. Preformatted data is available online under Data folder (*Coming Soon*)

## Dependencies Installation
### Determine your operating system (Linux/OS or Windows), Denoising environment will automatically be created

**Windows**:
1. Download the [`Windows_environment.yml`](Windows_environment.yml) from the repository
2. Change directory to where the environment is saved
3. Create a new enviroment using by running the following command in your Terminal:
    
    ```posh
    conda env create -f Windows_environment.yml
    ```
4. Activate new enviroment:
    ```posh
    conda activate Denoising
    ``` 
**Linux/OS**:
1. Download the [`Linux_environment.yml`](Linux_environment.yml) from the repository
2. Change directory to where the environment is saved
3. Create a new enviroment using by running the following command in your Terminal:
    
    ```posh
    conda env create -f Linux_environment.yml
    ```
4. Activate new enviroment:
    ```posh
    conda activate Denoising
    ``` 
## Training and Eval
**09/22/2022 Update:** We have added an option for using .json config files to specfiy your run parameters. Additionally, there is now support for hybrid .json and argparsing to make it easier to modify your config for evaluation vs. training without having multiple config files.

To train any model, you have several options to specify the run options:
1. Call all args and utilize argparsing (old implementation):
    ```posh
    python -u main.py train *model_architecture*  "*Model_Name*" cwd=*cwd* nadh_data=*nadh_data.npz* loss=*Loss_fcn* wavelet_function=*Wavelet_transform* loss_alpha=*SSIM contributation*
    ```
2. Write a .json config file with all desired parameters:
    ```posh
    python -u main.py config.json
    ```
3. Hybrid Argparsing and config file (*Used mostly for eval but can be used for training*):
    ```posh
    python -u main.py config.json mode=eval fad_data=*fad_data.npz* nadh_data=*nadh_data.npz* train_mode=0
    ```

**08/23-09/22 ArgParsing ONLY:**

To train any model, you simply need to specify the run options and call the [main.py](main.py) function:

    python -u main.py train *model_architecture*  "*Model_Name*" cwd=*cwd* nadh_data=*nadh_data.npz* loss=*Loss_fcn* wavelet_function=*Wavelet_transform* loss_alpha=*SSIM contributation*

To evaluate any model, you simply need to specify the same training runs options and call the [main.py](main.py) function with eval mode on:

    python -u main.py eval *model_architecture*  "*Model_Name*" cwd=*cwd* nadh_data=*nadh_data.npz* loss=*Loss_fcn* wavelet_function=*Wavelet_transform* loss_alpha=*SSIM contributation*

Available options include:
- `Mode` (string):
    - *train* - used when training a new model
    - *eval* - used to evaluate with a set of trained weights. Must have `weights_final.hdf5` saved in a folder called *Model_Name*.
- `model_architecture` (string):
    - You **must** specify which model you are trying to train
    - *rcan*
    - *care*
    - *wunet*
    - *resnet*
    - *srgan* (NOTE: SRGAN uses CARE for Content-Loss calculations instead of VGG19, please ensure you have CARE weights saved as **CARE_Pretrained.hdf5** in the path)
    - [`CARE_Pretrained.hdf5`](CARE_Pretrained.hdf5) weights for the CARE architecture used here are available on this repository
- `Model_Name` (string):
    - Specify the name of the model you are training. Good notion includes information about specific parameters used i.e.: 'NADH_CAREModel_SSIMR2Loss_alpha_p5_wavelet_bior1p1'
    - Here, we include what type of data the model will be trained on (NADH or FAD), which model (CARE, WUNET, SRGAN, or RCAN), which loss function (see below), the alpha level of each weight included, if wavelet tranform will be used (use *wunet*) and if so which type, and which training/validation seed will be used
- `cwd` (string):
    - List the path where the data (.npz) files are stored for training **(default = ' ')**
- `nadh_data` (string):
    - List the name of the NADH data file during training for NADH training **(default = ' ')**
- `fad_data` (string):
    - List the name of the FAD data file during training for FAD training **(default = ' ')**
- `epochs` (integer):
    - Maximum number of epochs desired **(default = 300)**
- `steps_per_epoch` (integer):
    - Number of steps per epoch, this controls the batch size of the model **(default = None for rcan, resnet, srgan or 100 for care)**
- `input_shape` (array of integers):
    - Set to the patch size used for your images **(default = [256 256])**
- `initial_learning_rate` (integer):
    - Starting learning rate used for training **(default = 1e-5)**
- `val_seed` (integer):
    - The random seed used to split training and validation datasets **(default = 0)**
- `val_split` (integer):
    - Sets the number of ROIs to include in the validation set  **(default = 4)**
    - It is important to note if `test_flag` = 0, all testing images will be added to `val_split`
- `test_flag` (integer):   
    - Controls whether a test set needs to be generated, if 0, a test set will be generated from the training data, if 1, the user is indicating they have a pre-formatted test set  **(default = 1)**
- `train_mode` (integer):   
    - Controls whether a test set needs to be extracted from the previous training dataset **(default = 0)**
    - A value of 0 means `test_flag`=1 and all data is new to the model 
    - A value of 1 means some data was used in training, in which case, set the `val_seed` and `val_split`, to the same values used during training so the designated test set can be extracted
- `ssim_FSize` (integer):   
    - Controls the filter size used by the SSIM function **(default = 11)**
- `ssim_FSig` (integer):   
    - Controls the filter sigma used by the SSIM function **(default = 1.5)**        
- `loss` (string):
    - Multiple options for loss functions exist:
        - *ssim_loss*
        - *ssiml1_loss* **(default)**
        - *ssimr2_loss*
        - *MSSSIM_loss* **(must preinitalize some weights)**
        - *pcc_loss*
        - *ssimpcc_loss*
        - *ffloss*
        - *SSIMFFL*
- `metrics` (array of strings): 
    - Multiple options for metrics exist:
        - *psnr*
        - *ssim*
        - *pcc* 
    - **Default: ['psnr', 'ssim']**
- `loss_alpha` (integer):
    - Controls how much different loss functions are weighted in the compound loss **(Default = 0.5)**
- `wavelet_function` (string):
    - If analysis included wavelet transformation of the data, specify which mother wavelet to use
        - chose from `pywt.wavelist()`
    - **Default = ' '**
- `val_seed` (integer):
    - Controls which ROIs are included in Training and Validation 
    - **Default = 0**
### Model Specfic Config:
**RCAN config**
- `num_channels` (integer): 
    - Number of channels for expansion **(Default = 32)**
- `num_residual_blocks` (integer):
    - Number of residual blocks in each residual group **(Default = 5)**
- `num_residual_groups` (integer):
    - Number of residual groups **(Default = 5)**
- `channel_reduction` (integer): 
    - Specifies the channel reduction factor which defines the number of filters used (`num_channels`/`channel_reduction` = `num_filters`) **(Default  = 4)**

**CARE UNET and WU-net config**
- `unet_n_depth` (integer):
    - Number of downsampling steps required before upsampling begins **(Default = 6)**
- `unet_n_first` (integer):
    - Number of channels for expansion **(Default = 32)**
- `unet_kern_size` (integer):
    - Kernel size used during convolution steps **(Default = 3)**

## References
<a id="1">[1]</a>
Chen, J., Sasaki, H., Lai, H. *et al.* Three-dimensional residual channel attention networks denoise and sharpen fluorescence microscopy image volumes. *Nat Methods* **18**, 678-687 (2021). https://doi.org/10.1038/s41592-021-01155-x

<a id="2">[2]</a>
Weigert, M., Schmidt, U., Boothe, T. *et al.* Content-aware image restoration: pushing the limits of fluorescence microscopy. *Nat Methods* **15**, 1090–1097 (2018). https://doi.org/10.1038/s41592-018-0216-7

<a id="3">[3]</a>
Ledig, C., Theis, L., Huszar, F. *et al.* Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. *arXiv* (2016). https://doi.org/10.48550/arxiv.1609.04802

<a id="4">[4]</a>
Aytekin, C., Alenius, S., Paliy, D. & Gren, J. A Sub-band Approach to Deep Denoising Wavelet Networks and a Frequency-adaptive Loss for Perceptual Quality. in *IEEE 23rd International Workshop on Multimedia Signal Processing*, MMSP 2021 1–6 (IEEE, 2021). https://doi.org/10.1109/MMSP53017.2021.9733576.

<a id="5">[5]</a>
K.B., SRGAN: Super Resolution Generative Adversarial Networks. *PaperspaceBlog* (2021). https://blog.paperspace.com/super-resolution-generative-adversarial-networks/. Date Accessed: 08/30/2022

<a id="6">[6]</a>
Laihong, J., Image Super-Resolution using SRResNet and SRGAN, (2021), GitHub repository, https://github.com/jlaihong/image-super-resolution