#Importing the necessary libraries
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
from PIL import Image # to make an image from an array
from IPython import display 

"""

A widely-used chart for illustrating frequency distributions is the histogram. 
In Histograms of Oriented Gradients (HOG), the gradients are captured and recorded within a 1D histogram.
"""
#use pyplot to plot the histogram
values = [1.1, 1.5, 2.2, 3.5, 3.5, 3.6, 4.1]
plt.hist(values, bins=4, range=(1,5))
plt.show()

"""
Each element in an array may carry an associated weight. 
The default weight is 1 for each element, indicating its contribution to the bin when included in the histogram. 
Utilize both the array values and their respective weights to generate weighted histograms.
"""

#plot the weighted histograms
values = [1.1, 1.5, 2.2, 3.5, 3.5, 3.6, 4.1]
weights = [1., 1., 3., 1.2, 1.4, 1.1, 0.2]
plt.hist(values, bins=4, range=(1,5), weights=weights)
plt.show()


"""
Consider an image, X, as an example. 
Given the potential presence of multiple pedestrians within X, the image is partitioned into smaller windows, commonly referred to as cells. 
Subsequently, the system analyzes each window, calculating the histogram of oriented gradients for each.

Now, let's focus on a specific window, referring to it as a cell.
A cell is essentially an array of numerical values, and we will leverage this cell for subsequent processing in the following steps.
"""
#prepare the image to demonstrate HOG
cell = np.array([
    [0, 1, 2, 5, 5, 5, 5, 5],
    [0, 0, 1, 4, 4, 5, 5, 5],
    [0, 0, 1, 3, 4, 5, 5, 5],
    [0, 0, 0, 1, 2, 3, 5, 5],
    [0, 0, 0, 0, 1, 2, 5, 5],
    [0, 0, 0, 0, 0, 1, 3, 5],
    [0, 0, 0, 0, 0, 0, 2, 5],
    [0, 0, 0, 0, 0, 0, 1, 3],
    ],dtype='float64')
plt.imshow(cell, cmap='binary', origin='lower')
plt.show()



"""
Calculate the gradients in both the x and y directions using the Sobel function from the cv2 library. 
Afterward, determine the magnitude and angle of the computed gradients.
"""

#calculate the histograms of oriented gradients
gradx = cv2.Sobel(cell, cv2.CV_64F,1,0,ksize=1)
grady = cv2.Sobel(cell, cv2.CV_64F,0,1,ksize=1)

norm, angle = cv2.cartToPolar(gradx,grady,angleInDegrees=True)


#Show the magnitude of the gradients and overlay an arrow indicating both the magnitude and direction of the gradient.

#plotting the norm and showing the magnitude

plt.imshow(norm, cmap='binary', origin='lower')

q = plt.quiver(gradx, grady, color='blue')
plt.show()

"""
Consequently, the peak on the right side corresponds to the visible edge of the cell. 
This histogram now characterizes a singular cell. 
To ascertain if the current detection window encompasses a human, we require a procedure that evaluates all cells within the window.
"""

#Plot HOG values using pyplot
plt.hist(angle.reshape(-1), weights=norm.reshape(-1), bins=20, range=(0,360))
plt.show()

"""
Every cell within the detection window possesses its individual Histogram of Oriented Gradients (HOG). 
These HOGs are consolidated into an extensive array of numerical values. 
For instance, if the detection window comprises 8x16 cells (equivalent to 128 cells) and each HOG consists of 20 bins, the total number of values amounts to 2560.
However, due to the high dimensionality (2560 in our illustration), a decision was made to employ a linear Support Vector Machine (SVM) in the OpenCV implementation of HOG.
"""
#load the HOG based human detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())



#load the video from given path and read a frame using cv
videoPath = 'video.mp4'
cap = cv2.VideoCapture(videoPath)

ret= True
#read frame-by-frame
ret, frame = cap.read()

#detect the pedestrian 
# resizing for faster detection
frame = cv2.resize(frame, (640, 480))
# using a greyscale picture, also for faster detection
gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
print("Debugging...")
# detect people in the image
# returns the bounding boxes for the detected objects
boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )

#draw the boxes on the image and visualize it
boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

for (xA, yA, xB, yB) in boxes:
    # display the detected boxes in the colour picture
    cv2.rectangle(frame, (xA, yA), (xB, yB),(0, 255, 0), 2)
frame = Image.fromarray(frame)
plt.imshow(frame)

print("Debugging 2.0....")

#examine the whole video for pedestrians
while(ret):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break  # or handle the failure in an appropriate way

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
    frame = Image.fromarray(frame)
    plt.imshow(frame)
    display.clear_output(wait=True)
    display.display(plt.gcf())

# When everything is done, release the capture
cap.release()