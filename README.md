# Gender-and-Age-detection
Project to detect Gender and Age from webcam, youtube video, and image stills
(Skip to end for implementation details)

A simple implementation of an Object detection model that can recognize human facial features and based on the same features, predict the gender and age of the subject. The facial detection is based on a Cascade classifier. The gender and age detector is made with a CNN architecture. In this project we test the detector for three different input scenarios, Webcam, input image, and youtube CCTV footage. All 3 models can be made with slight variations to a base code.

ENVIRONMENT USED

•	Spyder Python IDE (Python 3.7)

•	Anaconda environment

•	No GPU, Cuda

•	CPU computation only


LIBRARIES USED

•	OpenCV (cv2)

•	Matplotlib

•	Pafy

•	Youtube-dl


For this model we used a pre-trained classifier and CNN.
The Haar feature-based Cascade classifier is a pre trained classifier for OpenCV. 
The CNN and neural network are also pre-trained networks. There are 3 different implementations of the model in this project depending on the type of input dataset.

The 3 input datasets are:

•	Laptop Webcam input

•	MUCT Database:    3755 male and female faces with 76 manual landmarks. The database is augmented with variance of lighting, age, and ethnicity. https://github.com/StephenMilborrow/muct

•	Youtube video: “SharpView CCTV in Shopping Mall”   url: https://www.youtube.com/watch?v=SvldnZ6qMGU



BASIC WORKING OF THE MODEL

This model can be configured to derive input from either the webcam of the user or by directly providing test images/videos. The imported cv2 library performs multiple functionalities like pre-processing and output visualization tools.                                                                                                            
The Haar cascade takes the pre-processed input image and passes it through its classifier. First through the Facial classifier and after determining Region of Interest, the Eye classifier. 
After procuring the frame of the ROI the initial bounding box is placed using the coordinates received from the classifier.     
The frame of the ROI is then passed to the gender CNN network which then provides a prediction of the gender of the subject and the age CNN network which then provides a prediction of the range of the age of the subject. The prediction is then visualized over the bounding box of the detection.



IMPLEMENTATION

To implement this model for the different types of input simply download all the necessary pre-trained files from this repo
They are:
deploy_age.prototxt
deploy_gender.prototxt
gender_age_detection_imagestill.py
gender_age_detection_inputimage.py
gender_age_detection_youtube.py
haarcascade_eye.xml
haarcascade_frontalface_default.xml

Additionally you must download the following:
age_net.caffemodel
url: https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel

gender_net.caffemodel
url: https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel

Keep all of these files in your execution environment,

and simply exceute the pythong script!

Happy Learning!



