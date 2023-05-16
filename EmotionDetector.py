import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib
import os
import cv2
import time


def main():

    print("time start")
    start = time.time()
    images = []
    labels = []

    # Load Dataset
    training_dir = "train"  # Directory to the folders of all of the training data
    emotions = os.listdir(training_dir)

    print("loading dataset")
    for emotion in emotions:
        emotion_dir = os.path.join(training_dir, emotion)

        image_names = os.listdir(emotion_dir)
        emotion_image_paths = [os.path.join(
            emotion_dir, image_name) for image_name in image_names]

        images.extend(emotion_image_paths)
        labels.extend([emotion] * len(emotion_image_paths))

    unique_labels = np.unique(labels)
    print("Unique Emotion Labels:", unique_labels)
    # Extract Featres using SIFT and K-mean Clustering (Codebook)
    features = []

    print("extracting features")
    for image in images:
        img = cv2.imread(image)
        desc = extract_sift_features(img)
        if desc is not None:

            features.extend(desc)

    print("k clustering")

    k = 100
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(features)
    codebook = kmeans.cluster_centers_

    hist_feat = []
    print("creating histogram")
    for image in images:
        img = cv2.imread(image)
        desc = extract_sift_features(img)
        histogram = np.zeros(k)
        if desc is not None:
            for descriptor in desc:
                distances = np.linalg.norm(codebook - descriptor, axis=1)
                closest_cluster_index = np.argmin(distances)
                histogram[closest_cluster_index] += 1
        hist_feat.append(histogram)

    # Display Histogram
    histogram = np.mean(hist_feat, axis=0)  # Calculate the average histogram
    plt.bar(range(k), histogram)
    plt.xlabel("Cluster")
    plt.ylabel("Frequency")
    plt.title("Histogram of Visual Words")
    plt.show()

    X = np.array(hist_feat)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train the Model
    print("in svc")
    model = SVC()
    model.fit(X_train, y_train)

    # Test the model with itself
    print("testing")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    end = time.time()
    print(end-start)

    # dumping the training model as a .sav because this takes a looong time
    filename = "emotion_model.sav"
    codebook_filename = "emotion_model_codebook.sav"
    joblib.dump(model, filename)
    joblib.dump(codebook, codebook_filename)


def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return descriptors


if __name__ == "__main__":
    main()
