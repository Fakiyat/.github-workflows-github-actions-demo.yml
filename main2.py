import random
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math

# Capture the video from the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set the width of the frame
cap.set(4, 720)  # Set the height of the frame

# Initialize the hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)


class SnakeGame:
    def __init__(self, path_food):
        self.points = []  # List to store points of the snake
        self.lengths = []  # List to store the distance between each point
        self.current_length = 0  # Total length of the snake
        self.allowed_length = 150  # Length allowed for the snake
        self.previous_head = None  # Previous head of the snake

        # Load the food image
        self.img_food = cv2.imread(path_food, cv2.IMREAD_UNCHANGED)
        self.h_food, self.w_food, _ = self.img_food.shape  # Height and width of food
        self.food_point = 0, 0  # Position of the food
        self.random_food_location()  # Set a random food location

        self.score = 0  # Initialize score
        self.game_over = False  # Game over flag

    def random_food_location(self):
        # Randomize food location within the frame's bounds
        self.food_point = random.randint(100, 1000), random.randint(100, 600)

    def update(self, img_main, current_head):
        if self.previous_head is None:
            self.previous_head = current_head

        if self.game_over:
            # Display 'Game Over' message and score
            cvzone.putTextRect(img_main, "Game Over", [300, 400], scale=7, thickness=3, offset=20)
            cvzone.putTextRect(img_main, f'Your Score: {self.score}', [300, 550], scale=7, thickness=3, offset=20)
            cvzone.putTextRect(img_main, "Press 'r' to Restart", [300, 700], scale=7, thickness=3, offset=20)
        else:
            # Move the snake
            px, py = self.previous_head
            cx, cy = current_head

            self.points.append([cx, cy])
            distance = math.hypot(cx - px, cy - py)
            self.lengths.append(distance)
            self.current_length += distance
            self.previous_head = cx, cy

            # Length reduction if the snake exceeds the allowed length
            if self.current_length > self.allowed_length:
                for i, length in enumerate(self.lengths):
                    self.current_length -= length
                    self.lengths.pop(i)
                    self.points.pop(i)
                    if self.current_length < self.allowed_length:
                        break

            # Check if the snake head is at the food location
            rx, ry = self.food_point
            if rx - self.w_food // 2 < cx < rx + self.w_food // 2 and \
                    ry - self.h_food // 2 < cy < ry + self.h_food // 2:
                self.random_food_location()  # Generate new food location
                self.allowed_length += 50  # Increase snake length
                self.score += 1  # Increment score
                print(self.score)

            # Draw the snake
            if self.points:
                for i, point in enumerate(self.points):
                    if i != 0:
                        cv2.line(img_main, self.points[i - 1], self.points[i], (0, 0, 255), 20)
                cv2.circle(img_main, self.points[-1], 20, (0, 255, 0), cv2.FILLED)

            # Overlay the food image
            img_main = cvzone.overlayPNG(img_main, self.img_food, (rx - self.w_food // 2, ry - self.h_food // 2))

            # Display the score
            cvzone.putTextRect(img_main, f'Score: {self.score}', [50, 80], scale=3, thickness=3, offset=10)

            # Check for self-collision
            pts = np.array(self.points[:-2], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img_main, [pts], False, (0, 255, 0), 3)
            min_dist = cv2.pointPolygonTest(pts, (cx, cy), True)

            if -1 <= min_dist <= 1:
                print("Hit")
                self.game_over = True
                self.points = []  # Reset snake points
                self.lengths = []  # Reset snake lengths
                self.current_length = 0  # Reset snake length
                self.allowed_length = 150  # Reset allowed length
                self.previous_head = None  # Reset previous head position
                self.random_food_location()  # Reset food location

        return img_main


# Initialize the game with the food image
game = SnakeGame("Donut.png")

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip the image horizontally for a mirror effect
    hands, img = detector.findHands(img, flipType=False)  # Detect hands

    if hands:
        # Get the position of the index finger
        lm_list = hands[0]['lmList']
        point_index = lm_list[8][0:2]  # Coordinates of the index finger tip

        # Update the game with the new hand position
        img = game.update(img, point_index)

    # Show the game image
    cv2.imshow("Image", img)

    # Key press events
    key = cv2.waitKey(1)
    if key == ord('r'):
        game.game_over = False  # Reset the game when 'r' is pressed
