import cv2
from alright import WhatsApp
import tkinter as tk
from PIL import Image, ImageTk
from emojis_recognizer import EmojisRecognizer

EMOJIS_DIR = 'emoji_data/emoji'
emoji_index_to_path = {0: '1.png', 1: '2.png', 2: '3.png', 3: '4.png', 4: '5.png'}
emoji_index_to_sign = {0: '👆', 1: '✌', 2: '👌', 3: '🤘', 4: '🖖'}


def recognition(frame):
    """function for recognition of the hand emoji class form the frame"""
    # Creating emojis recognizer object
    emojis_recognizer = EmojisRecognizer(frame)
    # Finding the details of the hands in the frame
    hands_details = emojis_recognizer.find_hands_details()
    # Check if the frame contains hands
    if len(hands_details) == 0:
        return False, []
    # Converting the original image to hand images
    hand_images = emojis_recognizer.img2hands(hands_details)
    # Predicting the hand class of the hand images
    predictions = emojis_recognizer.predict_hand_class(hand_images, hands_details)
    # Converting the predictions to the corresponding emojis images
    emojis_recognizer.predictions2emojis(predictions)
    return True, predictions


def send(emoji_index, emoji_page):
    """function for sending the emoji to the desired contact"""
    # Find the emoji and send it to the contact
    emoji = EMOJIS_DIR + f'/{emoji_index_to_path[emoji_index]}'
    messenger.send_picture(emoji, 'emoji')
    # Give feedback on the sending operation to the user
    text_success = tk.Label(emoji_page)
    text_success.configure(text=f'successfully sent {emoji_index_to_sign[emoji_index]}', font=('Times', 16))
    text_success.grid()


def emojiWindow():
    """function for creating the page for emojis finding"""
    # Create the page for emojis sending
    emoji_page = tk.Tk()
    emoji_page.title('Hands Emojis')

    # Create a frame
    app = tk.Frame(emoji_page)
    app.grid()
    # create a label in the frame
    lmain = tk.Label(app)
    lmain.grid()

    # Open the camera
    cap = cv2.VideoCapture(0)

    # function for video streaming
    def video_stream():
        try:
            success, frame = cap.read()

            global emoji_index

            # Find the hand and predict the class
            hand_exists, predictions = recognition(frame)
            if len(predictions) > 0:
                # the emoji class is the predicted class
                emoji_index = predictions[0]
            else:
                # the emoji class is the default classes
                emoji_index = 0

            # Display the frame on the page
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)

            # Get more frames from the camera
            lmain.after(1, video_stream)

        except Exception:
            print('ERROR')

    # Add button for sending the emoji
    send_button = tk.Button(emoji_page, text="send", command=lambda: send(emoji_index, emoji_page),
                            font=('Times', 20), height=1, width=3, bg='purple', fg='white')
    send_button.grid()

    video_stream()

    emoji_page.mainloop()


def findName():
    """function for finding the user name in whatsapp and opening the page for emojis sending"""
    messenger.find_by_username(entryName.get())
    opening.destroy()
    emojiWindow()


# The background structure
try:
    messenger = WhatsApp()
except:
    pass

input("Press ENTER after login into Whatsapp Web and your chats are visible.\n\n")

# Create object
opening = tk.Tk()
opening.title('Hands Emojis')

# Add image file
image = Image.open("openingGUI.png")
resize_image = image.resize((400, 600))
img = ImageTk.PhotoImage(resize_image)

# Create Canvas
canvas1 = tk.Canvas(opening, width=400, height=600)
canvas1.pack(fill="both", expand=True)

# Display image
canvas1.create_image(0, 0, image=img, anchor="nw")

# Create Buttons
entryName = tk.Entry(opening, font=("arial", 20))
entryName_canvas = canvas1.create_window(202, 300, width=170, height=50, window=entryName)
searchButton = tk.Button(opening, text="Search", command=findName, font=('Times', 20),
                         height=2, width=3, bg='purple', fg='white')
searchButton_canvas = canvas1.create_window(127, 375, width=150, height=50, anchor="nw", window=searchButton)


while entryName.get() == 0:
    pass

# Execute tkinter
opening.mainloop()
