# POI-GAN

This the implementation of POI_GAN, which is multi-task learning based POI recommendation simultaneously considering the temporal check-ins and the geographical locations.

### Pre-request

> linux 16.04
>
> python 3.6
>
> PyTorch 1.0.1
>
> CUDA Version: 10.1

### Usage

- Clone the repo:  https://github.com/coderxor/POI-GAN.git

- mkdir  'experiment' to save checkpoint, parameters, result.
- Prepare your train/Val/test data and preprocess the data.
-  Refer to the codes of corresponding sections for specific purposes.

#### Dataset

- Foursquare <https://sites.google.com/site/yangdingqi/home/foursquare-dataset>
- Gowalla <https://snap.stanford.edu/data/loc-Gowalla.html>

#### Model

- Load.data.py

  > The code to define the Dataloader to load the POI dataset.

- model.py

  > The codes of the models in this paper, including the define and operation of the Generator, Discriminator.

#### Train&&Test

- POI_GAN_train.py

  > The codes to train the POI_GAN and evaluate the network on the test data.

- the Class  'POI_Recommendation_Experiment' aims to complete the train and test.
- the para_meta is the model *Hyperparameters* 
