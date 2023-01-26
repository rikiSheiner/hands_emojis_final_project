import os
import cv2

"""
Utils Module
"""


def count_fingers_up(fingers):
    """function to count how many fingers in the hand are up"""
    count = 0
    for i in range(len(fingers)):
        if fingers[i] == 1:
            count = count + 1
    return count


def predict_by_fingers(fingers):
    """function to detect the hand emoji class according to the indexes of raised fingers"""
    num_fingers_up = count_fingers_up(fingers)

    if num_fingers_up == 5:
        return 4
    elif num_fingers_up == 2:
        if (fingers[2] == 1 and fingers[3]) == 1 or (fingers[1] == 1 and fingers[2]):
            return 1
        elif (fingers[0] == 1 and fingers[3]) == 1 or (fingers[1] == 1 and fingers[4]):
            return 3
        elif (fingers[0] == 1 and fingers[1]) == 1 or (fingers[3] == 1 and fingers[4]):
            return 0

    return -1


def load_emojis(emojis_dir):
    """
    This function is used for loading the emojis images of the classes
    :param emojis_dir: the directory of the hand emogi images
    :return: array of images, masks and classes
    """
    size = 100
    data = []
    # loop for loading the emojis
    for img in os.listdir(emojis_dir):
        try:
            # Read emoji and resize
            emoji = cv2.imread(os.path.join(emojis_dir, img))
            emoji = cv2.resize(emoji, (size, size))
            # Create a mask of emoji
            img2gray = cv2.cvtColor(emoji, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
            # Add image, mask and class of the current emoji to the list
            data.append([mask, emoji, img.split('.')[0]])
        except Exception as e:
            pass

    return data
