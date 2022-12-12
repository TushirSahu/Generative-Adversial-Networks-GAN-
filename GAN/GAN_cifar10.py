import tensorflow as tf
import numpy as np
from numpy import reshape
import matplotlib.pyplot as plt
from keras.layers import Dense,Conv2D,Conv2DTranspose,LeakyReLU,Flatten,Reshape,Dropout
from keras.models import Sequential
from keras.optimizers import Adam


from keras.datasets.cifar10 import load_data
(X_train,Y_train),(X_test,Y_test)=load_data()

for i in range(25):
    plt.subplot(5,5,1+i)
    plt.imshow(X_train[i])
    
plt.show()
        

input_shape=(32,32,3)
def discriminator(in_shape=input_shape):
     model=Sequential()
     model.add(Conv2D(128,(3,3),strides=(2,2),padding='same',input_shape=in_shape))
     model.add(LeakyReLU(alpha=0.2))
     model.add(Conv2D(128,(3,3),strides=(2,2),padding='same',input_shape=in_shape))
     model.add(LeakyReLU(alpha=0.2))
     
     model.add(Flatten())
     model.add(Dropout(0.2))
     model.add(Dense(1,activation='sigmoid'))
     opt=Adam(lr=0.002,beta_1=0.5)
     model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
     return model
     


discr_model=discriminator()
print(discr_model.summary())

def generator(latent_dims):
   
    model = Sequential()
    n_nodes=128*8*8

    model.add(Dense(n_nodes, input_dim=latent_dims))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((8,8,128)))
    model.add(Conv2DTranspose(128, kernel_size=4, strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2DTranspose(128, kernel_size=4, strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(3,(8,8),activation='tanh',padding='same'))
    return model


gen_model=generator(100)
print(gen_model.summary())


def define_gan(discriminator,generator):
    discriminator.trainable=False
    
    model=Sequential()
    model.add(generator)
    model.add(discriminator)
    opt=Adam(lr=0.002,beta_1=0.5)
    model.compile(loss='binary_crossentropy',optimizer=opt)
    return model


def load_real():
    (X_train,_),(_,_)=load_data()
    X=X_train.astype('float32')    
    X=(X-127.5)/127.5
    return X

def generate_real_sample(dataset,n_samples):
    ix=np.random.randint(0,dataset.shape[0],n_samples)
    X=dataset[ix]
    y=np.ones((n_samples,1))
    return X,y

def generate_latent_points(latent_dims,n_samples):
    x_input=np.random.randn(latent_dims*n_samples)
    x_input=x_input.reshape(n_samples,latent_dims)
    return x_input



def generate_fake_samples(generator,latent_dims,n_samples):
    
    x_input=generate_latent_points(latent_dims, n_samples)    
    X=generator.predict(x_input)
    y=np.zeros((n_samples,1))
    return X,y




def train_model(g_model,d_model,gan_model,dataset,latent_dims,epochs=100,batch_size=128):
    bat_per_epo=int(dataset.shape[0]/batch_size)
    half_batch=int(batch_size/2)
    
    for i in range(epochs):
        for j in range(bat_per_epo):
            
            X_real,y_real=generate_real_sample(dataset, half_batch)
            
            d_loss_real,_=d_model.train_on_batch(X_real,y_real)
            
            X_fake,y_fake=generate_fake_samples(g_model, latent_dims, half_batch)
            d_loss_fake,_=d_model.train_on_batch(X_fake,y_fake)
            
            X_gan=generate_latent_points(latent_dims, batch_size)
            
            y_gan=np.ones((batch_size,1))
                
            g_loss=gan_model.train_on_batch(X_gan,y_gan)
            
            print('Epoch %d,Batch %d/%d,d1=%.3f,d2=%.3f g=%.3f'%(i+1,j+1,bat_per_epo,d_loss_real,d_loss_fake,g_loss))

    g_model.save('cifar_generator1.h5')




latent_dims=100

discr=discriminator()
gene=generator(latent_dims)
gan_model=define_gan(discr, gene)
dataset=load_real()




train_model(gene, discr, gan_model, dataset, latent_dims,epochs=2)
















    
    
    
    
    
    
    
    
    
    
    
    