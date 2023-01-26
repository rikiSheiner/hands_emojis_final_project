import cv2
import numpy as np

import utils
import hand_detector
from keras.models import load_model

"""
Emojis Recognizer Module

Author: Rivka Sheiner
"""

MODEL = load_model('hand_emoji_model')
EMOJIS_IMAGES = utils.load_emojis('emoji_data/emoji')


class EmojisRecognizer:
    def __init__(self, img, img_size=100, size=100, offset=20):
        self.img = img
        self.detector = hand_detector.HandDetector(detectionCon=0.8, maxHands=2)
        self.img_size = img_size
        self.size = size
        self.offset = offset
        self.model = MODEL
        self.emojis_images = EMOJIS_IMAGES

    def find_hands_details(self):
        hands = self.detector.findHands(self.img, draw=False)
        hands_details = {}
        for i in range(len(hands)):
            hands_details[i] = hands[i]
        return hands_details

    def img2hands(self, hands_details):
        img_size = self.img_size
        offset = self.offset
        gray_hand_images = []
        for i in range(len(hands_details)):
            # Bounding box info x,y,w,h
            x, y, w, h = hands_details[i]['bbox']
            # Cropping the hand from the original image
            img_crop = self.img[y - offset:y + offset + h, x - offset:x + offset + w]
            # Convert the hand image from BGR to grayscale mode
            gray_image = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            # Resizing the gray image
            img_gray_resize = cv2.resize(gray_image, (img_size, img_size))
            # Converting the gray image to array
            img_gray_resize = np.array(img_gray_resize).reshape(-1, img_size, img_size, 1)
            # Adding the gray hand image to the list of hands images
            gray_hand_images.append(img_gray_resize)
        return gray_hand_images

    def predict_hand_class(self, hand_images, hands_details):
        predictions = []
        for i in range(len(hand_images)):
            # Predicting the emoji according to the gray hand image
            prediction = self.model.predict_on_batch(hand_images[i])
            # Getting the class number of the hand emoji
            predicted_number = np.argmax(prediction)
            prediction_by_fingers = utils.predict_by_fingers(self.detector.fingersUp(hands_details[i]))
            if prediction_by_fingers > -1:
                predicted_number = prediction_by_fingers
            predictions.append(predicted_number)
        return predictions

    def predictions2emojis(self, predictions):
        for i in range(len(predictions)):
            predicted_number = predictions[i]
            # Finding the relevant hand emoji image and putting it on the original image
            if predicted_number in range(len(self.emojis_images)):
                # Region of Image (ROI), where we want to insert emoji image
                roi = self.img[-self.size-10 - 90*i:-10 - 90*i, -self.size-10-90*i:-10-90*i]
                mask = self.emojis_images[predicted_number][0]
                # Set an index of where the mask is
                roi[np.where(mask)] = 0
                # Putting the relevant emoji image on the original image
                roi += self.emojis_images[predicted_number][1]

