# Startup Project: GANsmania

## Project layout: 
In this project we will design a network of Deep learning models in order to learn the construct and charatorstics of abstract images and try to replicate them by generating new images that have inherited the essence of the abstract images we trained our models on.

- **Stage 1: Image scrapping** extracting images from open sources and online Galleries that will be the base of our training. So far we have around 10,000 images and with help of image augemtation tools we can double or tribble the number of images for training.

- **Stage 2: Training the model** on the images. this part happens on the part of the GANs network called the discriminator model. This model's job is to decode the each image for colors, shape, patters, shades and features that make abstract art as it is. After learning all features from the all the images the discriminator becomes an expert in identifying if images are labeled as abstract or not. In the next step we put our newly trained model up against a model that is specialized in drawing patterns and shapes, calle the generator. 

- **Stage 3: putting the two models against each other**. The generator model will randmoly draw and generate new images that would be labeled by the descriminator as non abstract at the beginning, but with lots of training time and enough diverse images there will be a breakthrough by the generator model where it will start generating images that will have enough features of abstract images which will trick the discriminator model into thinking it is a real abstract image.

- **Stage 4: generating new image by inputing random data points** if the training result was satisfactory and testing the output

- **Stage 5:  generating new image by inputing data points that has been taken or influnced by audio** (audio clip,  voice, or music). This can be done extracting certain features from an audio file and trying to influnce the output to be manifisted in that generated image.

- **Stage 6: Deploying the project to the cloud** and building an interface for it to be used by users. 

##Workflow: 

![work flow](https://user-images.githubusercontent.com/81450873/134007625-3a32b015-0157-453e-9122-071b800231ed.jpg)



## Tasks: 

-	Image scrapping: Creating some tool to scrape images from online abstract art gallery. Using beautifulsoup or selenium if needed. Aim for it is somehwere between 50,000 to 100,000 in order we can use them in a base model.
-	Sound (data points exploration and features): extracting features from music or audio files
-	Project build up.
-	Platform for training GC notebook
-	Model tuning, data preprocessing and optimizing
-	Interface/Streamlit


[Anton and GANs.pptx](https://github.com/AliSalem2/GANsmania/files/7195910/Anton.and.GANs.pptx)
