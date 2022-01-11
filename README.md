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
