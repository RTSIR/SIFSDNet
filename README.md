# SIFSDNet
This repo contains the KERAS implementation of "SIFSDNET: SHARP IMAGE FEATURE BASED SAR DENOISING NETWORK"

# Run Experiments

To test for SAR denoising using SIFSDNet write:

python Test_SAR.py

The resultant images will be stored in 'Test_Results/SAR/'

Image wise ENL for the whole image database will also be displayed in the console as output.

To test for synthetic denoising using SIFSDNet write:

python Test_Synthetic.py

The resultant images will be stored in 'Test_Results/Synthetic/'

Image wise PSNR & SSIM as well as Average PSNR & Average SSIM for the whole image database will also be displayed in the console as output.


# Train SIFSDNet denoising network

To train the SIFSDNet denoising network, first download the [UC Merced Land Use data](http://weegee.vision.ucmerced.edu/datasets/landuse.html) and copy the images into genData folder. Then generate the training data using:

python generateData.py

This will save the training patch 'img_clean_pats.npy' in the folder 'trainingPatch/'

Then run the SIFSDNet model file for synthetic image denoising using:

python SIFSDNet_Synthetic.py

This will save the 'SIFSDNet_Synthetic.h5' file in the folder 'Pretrained_models/'.

Then run the SIFSDNet model file for SAR image denoising using:

python SIFSDNet_SAR.py

This will save the 'SIFSDNet_SAR.h5' file in the folder 'Pretrained_models/'.

# Citation
@inproceedings{thakur2022sifsdnet,
  title={Sifsdnet: Sharp image feature based sar denoising network},
  author={Thakur, Ramesh Kumar and Maji, Suman Kumar},
  booktitle={IGARSS 2022-2022 IEEE International Geoscience and Remote Sensing Symposium},
  pages={3428--3431},
  year={2022},
  organization={IEEE}
}
