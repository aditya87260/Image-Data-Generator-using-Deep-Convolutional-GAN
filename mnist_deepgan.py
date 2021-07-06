from keras.datasets import mnist
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
import math

(X_Train,_),(_,_) = mnist.load_data()

print(X_Train.shape)
#(60000, 28, 28)

X_Train=X_Train.reshape((*(X_Train.shape),1))



# Normalize this data [-1,1] 
X_Train  = (X_Train.astype('float32') - 127.5)/127.5
print(np.min(X_Train))
print(np.max(X_Train))

print(X_Train.shape)

TOTAL_EPOCHS = 50
BATCH_SIZE = 256
NO_OF_BATCHES = math.ceil(X_Train.shape[0]/float(BATCH_SIZE))
HALF_BATCH = 128
NOISE_DIM = 100 # Upsample into 784 Dim Vector
adam = Adam(lr=2e-4,beta_1=0.5)

# Generator 
# Input Noise (100 dim) and Outputs a Vector (28*28*1 dim)
#Upsampling 
generator = Sequential()
generator.add(Dense(7*7*128,input_shape=(NOISE_DIM,))) #Upsampled noice vector not a 3d vector
generator.add(Reshape((7,7,128))) #here we have reshaped upsampled noice vector to a 3d vecctor of shape(7,7,128)
generator.add(LeakyReLU(0.2))
generator.add(BatchNormalization())


generator.add(UpSampling2D()) #Double the Size to 14 X 14 X 128
generator.add(Conv2D(64,kernel_size=(5,5),padding='same')) #Now reduce the no.of channels to 64 so new size of image is 14 X 14 X 64
generator.add(LeakyReLU(0.2))
generator.add(BatchNormalization())

# Double the  Size again and reduce channel to 1 so now new size of image is  28 X 28 X 1
generator.add(UpSampling2D())
generator.add(Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh'))

# Final Output (No ReLu or Batch Norm)
generator.compile(loss='binary_crossentropy', optimizer=adam)
generator.summary()


#Discriminator - Downsampling
discriminator = Sequential()  #firstly we will pass the fake and real images of shape 28 X 28 X 1
discriminator.add(Conv2D(64,(5,5),strides=(2,2),padding='same',input_shape=(28,28,1))) #Now we have change the shape to 14 X 14 X 64
discriminator.add(LeakyReLU(0.2))

# Prefer Stride Convolutions over MaxPooling
discriminator.add(Conv2D(128,(5,5),strides=(2,2),padding='same'))#Now we have change the shape to 7 X 7 X 128
discriminator.add(LeakyReLU(0.2))


discriminator.add(Flatten()) # now we have flatten the output and convert it to a singe vector of 6272 dim i.e. product of those numbers(7*7*128)
discriminator.add(Dense(1,activation='sigmoid'))

discriminator.compile(loss='binary_crossentropy',optimizer=adam)
discriminator.summary()

# GAN (Step-2)
discriminator.trainable = False
gan_input = Input(shape=(NOISE_DIM,))
generated_img = generator(gan_input)
gan_output = discriminator(generated_img)

# Functional API
model = Model(gan_input,gan_output)
model.compile(loss='binary_crossentropy',optimizer=adam)



def save_imgs(epoch,samples=100):
    
    noise = np.random.normal(0,1,size=(samples,NOISE_DIM)) #generate some random noice vector data of dimension equal to noice_dim and between 0 and 1 because we have normalize image data between 0 and 1
    generated_imgs = generator.predict(noise)  #Now we pass this noice vector in generator to generate some fake data
    generated_imgs = generated_imgs.reshape(samples,28,28)
    
    plt.figure(figsize=(10,10))
    for i in range(samples):
        plt.subplot(10,10,i+1)
        plt.imshow(generated_imgs[i],interpolation='nearest',cmap='gray')
        plt.axis("off")
        
    plt.tight_layout()
    plt.savefig('images/gan_output_epoch_{0}.png'.format(epoch+1))
    plt.show()

# Training Loop
d_losses = []     #Disciminator loss
g_losses = []     #Generator loss


for epoch in range(TOTAL_EPOCHS):
    epoch_d_loss = 0.  #  Initial loss we have assume as 0
    epoch_g_loss = 0.
    
    #Mini Batch SGD
    for step in range(NO_OF_BATCHES):
        
        # Step-1 Discriminator 
        #We will pass 50% Real Data + 50% Fake Data in discriminator
        
        #Real Data X
        idx = np.random.randint(0,X_Train.shape[0],HALF_BATCH) # Here we have generated some random numbers as indexes to generate some random images from training data for a particular batch of real images 
        real_imgs = X_Train[idx]
        
        #Fake Data X
        noise = np.random.normal(0,1,size=(HALF_BATCH,NOISE_DIM))  #Now we have generated some random noice vector value to get some fake data
        fake_imgs = generator.predict(noise) # Pass the random noice in generator to generate some fake data
        
        
        # Labels  # Now we have to genearte labels i.e. 0 for fake image and 1 for real image so that discimator learn the differnce between a real and a fake image 
        real_y = np.ones((HALF_BATCH,1))*0.9 #One Side Label Smoothing for Discriminator i.e. for real image we have consider the probability as 0.90 in place of 1
        fake_y = np.zeros((HALF_BATCH,1))
        
        # Train our Discriminator 
        # Now using train_on batch function we will pass real images, real label and fake image ,fake label in discriminator and it will return loss in both cases now we will take 50% of both loses to generate total loss and then and then we will calculate total loss for an epoch by adding the loss from each batch of images  
        d_loss_real = discriminator.train_on_batch(real_imgs,real_y)
        d_loss_fake = discriminator.train_on_batch(fake_imgs,fake_y)
        d_loss = 0.5*d_loss_real + 0.5*d_loss_fake
        
        epoch_d_loss += d_loss
        
        # Train Generator (Considering Frozen Discriminator)
        # now we will consider disciminator as frozen and train the generator so ww will pass the random noice vector and the the label but this time we will passing label as 1 in ground_truth_y because we want that discrimator should declare our fake images as 1 so we are attaching label as 1 with ur images now disciminator will declare it as fake or real acc. to its training
        noise = np.random.normal(0,1,size=(BATCH_SIZE,NOISE_DIM))
        ground_truth_y = np.ones((BATCH_SIZE,1))
        g_loss = model.train_on_batch(noise,ground_truth_y)
        epoch_g_loss += g_loss
        
    print("Epoch %d Disc Loss %.4f Generator Loss %.4f" %((epoch+1),epoch_d_loss/NO_OF_BATCHES,epoch_g_loss/NO_OF_BATCHES))
    d_losses.append(epoch_d_loss/NO_OF_BATCHES)
    g_losses.append(epoch_g_loss/NO_OF_BATCHES)
    
    if (epoch+1)%5==0:
        generator.save('model/gan_generator_{0}.h5'.format(epoch+1))
        save_imgs(epoch)


plt.plot(d_losses,label="Disc")
plt.plot(g_losses,label="Gen")
plt.legend()
plt.show()