# SPDNet-demo

This is the demo code for SPDNet.

### Some important configurations
* torch==0.4.1
* torchvision==0.2.1
* python==3.6
* numpy==1.16.4

### Downloading models and datasets
Before test, pretrained models and datasets are needed: 
* Downloading Pretrained models and SUN dataset:
  * https://drive.google.com/drive/folders/1Zjxrn6poBxdNjQBlYvVyneVS2mkiT6-r?usp=sharing
  * https://drive.google.com/drive/folders/1YoXlHBNx-ftyGm01UUXJfVyc9fPoIKtY?usp=sharing
* Copy pretrained models to ./save_models, then "netDec_0.4474.model" and "netG_0.4474.model" are included.
* Copy SUN dataset to ./data. Dataset is alos available [here](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/).

### Testing Demo 
Then run the command: 
* python ./image-scripts/run_sun_test.py

**Note that all information about the relevant accounts is anonymized.**

