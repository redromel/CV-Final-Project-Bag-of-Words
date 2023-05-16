# CV-Final-Project-Bag-of-Words

This Project uses Bag of Visual Word Techniques in order to detect the emotion of a person given an image of said person.  there are 2 files, EmotionDetector.py is the training program that used Bag of Words Techniques in order to build a model that can guess emotions.  emotionGuess.py takes an image and predicts the emotion of the person in that image.  The program predicts from 7 different emotions, angry, disgust, fear, happy, neutral, sad, and surprise

How to Use:

 1. Make sure to have numpy, scikit-learn, joblib, and cv2 libraries install on your device
 2. Run emotionGuess.py
 3. Enter path to input image when prompted
 4. emotionGuess.py will return the predicted emotion of the image.  

Alternatively if you have conda an 'environments.yml' file is provided (note, I was not able to test this with conda)
	 
*Note: The training dataset was not included in the repository because it was too big and had too many files, link to the dataset can be found [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)*  
