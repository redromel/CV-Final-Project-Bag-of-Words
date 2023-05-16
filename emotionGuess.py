import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import cv2
import time


def main():

    model = joblib.load("emotion_model.sav")
    codebook = joblib.load("emotion_model_codebook.sav")

    image_path = input("Enter path to input image:  ")

    start = time.time()

    image = cv2.imread(image_path)
    desc = extract_sift_features(image)

    k = 100
    histogram = np.zeros(k)
    if desc is not None:
        for descriptor in desc:
            distances = np.linalg.norm(codebook - descriptor, axis=1)
            closest_cluster_index = np.argmin(distances)
            histogram[closest_cluster_index] += 1

    X = np.reshape(histogram, (1, -1))

    predict = model.predict(X)

    end = time.time()
    print(end-start)

    title = 'Predicted emotion: ' + predict[0]
    print(title)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_sift_features(image):
    img_scale = cv2.resize(image, (48, 48))
    img_gray = cv2.cvtColor(img_scale, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)

    return descriptors


if __name__ == "__main__":
    main()