# Import Modules
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim


"""
Creating a transformation using torchvision.transforms: 
* Compose(): This method will be used to create a transformation.
* ToTensor(): This takes the image’s pixel values and scales them from 0–255 to 0–1.
* Normalize(): This method accepts the two parameters of mean and standard deviation and
normalizes the tensor with these values. You can use 0.5 for both of these parameters.
"""

# Create a Transformation
normalized = transforms.Normalize((0.5,), (0.5,))
tensor = transforms.ToTensor()
transformation = transforms.Compose([tensor, normalized])


"""
using built in dataset available through torchvision.datasets. this method accepts the following parameters:  
* root: The directory where it will download the ubyte files to load the datasets.

* download: A boolean parameter to specify whether data needs to be downloaded or not. 
If True, data will be downloaded to the root directory.

* train: A boolean parameter to choose between the training or testing data. 
If True, the function will load the training data. Otherwise, it will load the testing data.

* transform: A function to transform the dataset.
"""

# Download and Load Datasets
#training_dataset = datasets.MNIST('/bytefiles', download=True, train=True, transform=transformation)
#testing_dataset = datasets.MNIST('/bytefiles', download=True, train=False, transform=transformation)
import tempfile

# Create a temporary directory
temp_dir = tempfile.mkdtemp()
training_dataset = datasets.MNIST(temp_dir, download=True, train=True, transform=transformation)
testing_dataset = datasets.MNIST(temp_dir, download=True, train=False, transform=transformation)

train_data = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
test_data = torch.utils.data.DataLoader(testing_dataset, batch_size=64, shuffle=True)



"""
to be able to visualize the images from the datasets, we use iter() method to iterate over the training data 
and the imshow() method from matplotlib.pyplot to plot the images.
""" 

# Visualize Images
# images, labels = iter(train_data).next()
# for i in range(1, 31):
#     plt.subplot(3,10, i)
#     plt.subplots_adjust(wspace=0.3)
#     plt.imshow(images[i].numpy().squeeze())

# Get a batch of data from the DataLoader
data_iterator = iter(train_data)
images, labels = next(data_iterator)  # Use `next` to get the next batch

for i in range(1, 31):
    plt.subplot(3, 10, i)
    plt.subplots_adjust(wspace=0.3)
    plt.imshow(images[i].numpy().squeeze())

plt.show()

"""
To get the size for the input layer, you need to evaluate the size of the tensor. 
To get the tensor size, check the shape of any image from the images variable

"""    
print(images.shape)

# Calculate Size of Layers
input_layer = 784
hidden_layer1 = 64
hidden_layer2 = 32
output_layer = 10

"""
Creating the model to train using the Sequential() method from the torch.nn module, 
which wraps the layers into a network model  
* nn.Linear(): to map the input layer to the first hidden layer, 
the first hidden layer to the second hidden layer, and the second hidden layer to the output layer. 
It takes two integer values representing the sizes of the two layers, and connects them via a linear transformation.
* nn.ReLU(): the non-linear activation function after applying each nn.Linear map, except for the last one.
"""

# Build a Model
model = nn.Sequential(nn.Linear(input_layer, hidden_layer1),
nn.ReLU(),
nn.Linear(hidden_layer1,hidden_layer2),
nn.ReLU(),
nn.Linear(hidden_layer2, output_layer))

"""
Loss function is used to determine the accuracy of our model by defining the difference between the predictions by the model and the desired outcomes. 
The greater the disparity between the two, the greater the loss.  
To minimize the loss, the cross-entropy loss is used to adjust the model weights during training using backpropagation.
"""

# Calculate Cross-Entropy Loss
images = images.view(images.shape[0], -1)
outputs = model(images)
lossFunction = nn.CrossEntropyLoss()
loss = lossFunction(outputs, labels)

#Gradient descent will optimize the weights with each iteration of the model
# Obtain the Stochastic Gradient Descent Optimizer
gradient_descent = optim.SGD(model.parameters(), lr=0.1)

"""
* zero_grad() function of the gradient_descent variable to reset the weights to zero after iteration. This is to avoid weight-accumulation.

* Calculate the cross-entropy loss.

* Update the weights using backpropagation.

* Optimize the weights using gradient descent.

"""

# Train the Model
epochs = 20
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_data:
        images = images.view(images.shape[0], -1)
        # Feed-Forward
        gradient_descent.zero_grad()
        loss = lossFunction(model(images), labels)
        # Back Propagation
        loss.backward()
        # Optimize the weights
        gradient_descent.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(labels)
    print("Iteration : ", epoch+1, end = "\t")
    print("Loss: ", epoch_loss)

# Get the Predicted Label
def get_predicted_label(image):
    image = image.view(1, 28*28)
    with torch.no_grad():
        prediction_score = model(image)
    return np.argmax(prediction_score)

images, labels = next(iter(test_data))
print("Predicted Label: ", 
get_predicted_label(images[0]))
print("Actual Label: ", labels.numpy()[0])


# Test the Model
totalCount = 0
accurateCount = 0
for images, labels in test_data:
    for i in range(len(labels)):
        predictedLabel = get_predicted_label(images[i])
        actualLabel = labels.numpy()[i]
        print("Predicted Label: ", predictedLabel, " / Actual Label: ", actualLabel)
        if(predictedLabel == actualLabel):
            accurateCount += 1
    totalCount += len(labels)
print("Total images tested: : ", totalCount)
print("Accurate predictions: ", accurateCount)
print("Accuracy percentage: ", ((accurateCount/totalCount)*100), "%")