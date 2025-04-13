import sys
import signal
import time
import os  # Added for file path checking
from joblib import dump, load  # To get the model to the robot

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier

from mbot_bridge.api import MBot
from utils.camera import CameraHandler
from utils.robot import plan_to_pose, turn_to_theta
from waypoint_writer import read_labels_and_waypoints

# TODO: Update PATH_TO_MODEL.
PATH_TO_MODEL = "model.joblib"

robot = MBot()

def signal_handler(sig, frame):
    print("Stopping...")
    robot.stop()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

def main():
    ch = CameraHandler()  # Initialize the camera

    # Ensure the model file exists
    if not os.path.exists(PATH_TO_MODEL):
        print(f"Error: Model file {PATH_TO_MODEL} does not exist.")
        sys.exit(1)

    model = load(PATH_TO_MODEL)

    # Load waypoints and labels
    labels, waypoints = read_labels_and_waypoints()  # Load from waypoints.txt

    if not labels or not waypoints:
        print("Error: No labels or waypoints found. Please create or check waypoints.txt.")
        sys.exit(1)

    # Main loop to navigate through waypoints
    while True:
        frame = ch.get_processed_image()
        if frame is None:
            print("No post-it detected. Waiting...")
            time.sleep(1)
            continue

        # Predict the label from the processed image
        try:
            label = model.predict([frame.flatten()])[0]
            print(f"Detected label: {label}")
        except Exception as e:
            print(f"Error in model prediction: {e}")
            continue

        if label not in labels:
            print(f"Label {label} not found in waypoints. Ignoring and continuing...")
            continue

        # Get the corresponding waypoint
        waypoint_index = labels.index(label)
        goal_x, goal_y, goal_theta = waypoints[waypoint_index]

        print(f"Navigating to waypoint {label}: ({goal_x}, {goal_y}, {goal_theta})")

        # Navigate to the waypoint
        plan_to_pose(goal_x, goal_y, robot)
        turn_to_theta(goal_theta, robot)

        if label == 0:  # Return to start if label is 0
            print("Returning to start and exiting...")
            plan_to_pose(0, 0, robot)
            turn_to_theta(0, robot)
            break

if __name__ == '__main__':
    main()
