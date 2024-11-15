'''
  File name: objectTracking.py
  Author: Modified from Singh Sarvagya's and Shlok implementation
  Date created: 2024-11-07
  Description: Multi-face tracking using Fast-KLT with occlusion handling
'''
import cv2
import numpy as np
from fast_klt import FastKLT

class FaceTracker:
    def __init__(self, tracker_id, frame, bbox):
        # Initialize tracker parameters
        self.paramTuple = (1000, (10, 10), 10, 0.2, (15, 15), 3.0, tracker_id)
        self.tracker = FastKLT(self.paramTuple)
        self.tracker.initTracker(frame, bbox)
        self.bbox = bbox
        self.lost_frames = 0
        self.max_lost_frames = 30  # Maximum frames to keep trying to recover tracking
        self.confidence = 1.0

    def update(self, frame):
        success, new_bbox = self.tracker.updateTracker(frame)
        if success:
            self.bbox = tuple(map(int, new_bbox))
            self.lost_frames = 0
            self.confidence = 1.0
            return True, self.bbox
        else:
            self.lost_frames += 1
            self.confidence *= 0.8  # Reduce confidence when tracking is lost
            return False, self.bbox

    def is_tracking_lost(self):
        return self.lost_frames > self.max_lost_frames

def objectTracking(draw_bb=True, play_realtime=True, save_to_file=False):
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    # Initialize face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # List to store active face trackers
    active_trackers = []

    # Video writer setup
    if save_to_file:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width, frame_height))

    def detect_new_faces(frame, existing_bboxes):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        new_faces = []

        for face in faces:
            is_new = True
            face_center = np.array([face[0] + face[2]/2, face[1] + face[3]/2])

            # Check if this face overlaps with any existing tracked face
            for bbox in existing_bboxes:
                bbox_center = np.array([bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2])
                distance = np.linalg.norm(face_center - bbox_center)

                if distance < max(face[2], bbox[2]) * 0.5:  # If centers are close enough
                    is_new = False
                    break

            if is_new:
                new_faces.append(tuple(face))

        return new_faces

    frame_count = 0
    detection_interval = 15  # Detect new faces every N frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_count += 1
        current_bboxes = [tracker.bbox for tracker in active_trackers]

        # Periodic detection of new faces
        if frame_count % detection_interval == 0:
            new_faces = detect_new_faces(frame, current_bboxes)
            for i, face_bbox in enumerate(new_faces):
                tracker_id = len(active_trackers) + 1
                new_tracker = FaceTracker(tracker_id, frame, face_bbox)
                active_trackers.append(new_tracker)
                print(f"New face detected and tracker {tracker_id} initialized")

        # Update all trackers and handle lost tracks
        for tracker in active_trackers[:]:  # Use slice copy to allow removal during iteration
            success, bbox = tracker.update(frame)

            if success:
                if draw_bb:
                    x, y, w, h = bbox
                    # Draw with varying colors based on confidence
                    color = (0, int(255 * tracker.confidence), 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    # Add tracker ID
                    cv2.putText(frame, f"ID: {tracker.paramTuple[-1]}", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            elif tracker.is_tracking_lost():
                active_trackers.remove(tracker)
                print(f"Tracker {tracker.paramTuple[-1]} removed due to lost tracking")

        # Display tracking statistics
        cv2.putText(frame, f"Tracking {len(active_trackers)} faces", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show real-time tracking
        if play_realtime:
            cv2.imshow("Multi-Face Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Press 'q' to quit
                break
            elif key == ord('r'):  # Press 'r' to force redetection
                frame_count = 0

        # Save frame if requested
        if save_to_file:
            out.write(frame)

    # Cleanup
    cap.release()
    if save_to_file:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    objectTracking(draw_bb=True, play_realtime=True, save_to_file=True)



'''
import cv2
import numpy as np
from fast_klt import FastKLT


def objectTracking(draw_bb=True, play_realtime=True, save_to_file=False):
    # Attempt to open the webcam
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    # Face detection parameters
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    paramTuple = (1000, (10, 10), 10, 0.2, (15, 15), 3.0, 1)  # Increased keypoints and lowered FAST threshold

    # Lists to store trackers and bounding boxes
    trackers = []
    bounding_boxes = []

    # Initialize output video file if save_to_file is True
    if save_to_file:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))

    def initialize_trackers():
        """ Detect all faces in the current frame and initialize trackers for each. """
        trackers.clear()
        bounding_boxes.clear()
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Initialize a FastKLT tracker for each detected face
            tracker = FastKLT(paramTuple)
            tracker.initTracker(frame, (x, y, w, h))
            trackers.append(tracker)
            bounding_boxes.append((x, y, w, h))
            print("Initialized tracker for face with bounding box:", (x, y, w, h))

        return True

    # Initialize trackers for the first frame
    if not initialize_trackers():
        cap.release()
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Update each tracker independently
        for i, tracker in enumerate(trackers):
            success, rectOutput = tracker.updateTracker(frame)

            if success:
                # Update the bounding box with the tracked coordinates
                x, y, w, h = map(int, rectOutput)
                bounding_boxes[i] = (x, y, w, h)
                print(f"Tracker {i} bounding box coordinates:", (x, y, w, h))

                # Draw the bounding box on the frame
                if draw_bb:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                print(f"Tracking lost for tracker {i}. Reinitializing tracker...")
                initialize_trackers()  # Reinitialize all trackers if any tracker loses track
                break  # Restart the tracking loop after reinitialization

        # Show real-time tracking
        if play_realtime:
            cv2.imshow("Real-time Multi-Face Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break

        # Write to output file if saving
        if save_to_file:
            out.write(frame)

    # Release resources
    cap.release()
    if save_to_file:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    objectTracking(draw_bb=True, play_realtime=True, save_to_file=True)

'''
'''
import cv2
import numpy as np
from fast_klt import FastKLT


def objectTracking(draw_bb=True, play_realtime=True, save_to_file=False):
    # Attempt to open the webcam
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    # Face detection parameters
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    paramTuple = (1000, (10, 10), 10, 0.2, (15, 15), 3.0, 1)  # Increased keypoints and lowered FAST threshold
    tracker = FastKLT(paramTuple)

    # Initialize output video file if save_to_file is True
    if save_to_file:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))

    # Capture first frame and detect faces
    def initialize_tracker():
        while True:
            ret, frame1 = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                return None, None

            # Detect faces in the first frame
            gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                # Initialize the tracker on the first detected face
                (x, y, w, h) = faces[0]
                face_rect = (x, y, w, h)
                tracker.initTracker(frame1, face_rect)
                print("Face detected and tracker initialized with coordinates:", face_rect)
                return frame1, face_rect
            else:
                print("No faces detected in the initial frame. Trying again...")

    frame1, face_rect = initialize_tracker()
    if frame1 is None:
        cap.release()
        return

    while cap.isOpened():
        ret, frame2 = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Track the face using Fast-KLT tracker
        success, rectOutput = tracker.updateTracker(frame2)

        if success:
            # Draw the bounding box for the tracked face, converting to integer values
            x, y, w, h = map(int, rectOutput)
            print("Tracker bounding box coordinates:", (x, y, w, h))
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            print("Tracking lost or insufficient confidence. Reinitializing tracker...")
            frame1, face_rect = initialize_tracker()
            if frame1 is None:
                break  # Exit if reinitialization fails
            continue  # Skip this frame after reinitializing the tracker

        # Show real-time tracking
        if play_realtime:
            cv2.imshow("Real-time Tracking", frame2)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break

        # Write to output file if saving
        if save_to_file:
            out.write(frame2)

    # Release resources
    cap.release()
    if save_to_file:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    objectTracking(draw_bb=True, play_realtime=True, save_to_file=True)
'''
'''
import cv2
import numpy as np
from getFeatures import getFeatures
from estimateAllTranslation import estimateAllTranslation
from applyGeometricTransformation import applyGeometricTransformation


def objectTracking(draw_bb=False, play_realtime=True, save_to_file=False):
    cap = cv2.VideoCapture(0)  # Set up the webcam capture; "0" is typically the default for built-in cameras

    # Initialize the first frame and bounding boxes
    ret, frame1 = cap.read()
    if not ret:
        print("Failed to capture video from webcam.")
        cap.release()
        return

    if draw_bb:
        n_object = int(input("Number of objects to track:"))
        bbox = np.empty((n_object, 4, 2), dtype=float)
        for i in range(n_object):
            (xmin, ymin, boxw, boxh) = cv2.selectROI(f"Select Object {i}", frame1, False)
            cv2.destroyWindow(f"Select Object {i}")
            bbox[i, :, :] = np.array(
                [[xmin, ymin], [xmin + boxw, ymin], [xmin, ymin + boxh], [xmin + boxw, ymin + boxh]]).astype(float)
    else:
        # Default bounding box if no manual selection
        n_object = 1
        bbox = np.array([[[291, 187], [405, 187], [291, 267], [405, 267]]]).astype(float)

    # Prepare for output video file if saving to file
    if save_to_file:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))

    startXs, startYs = getFeatures(cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY), bbox)

    while cap.isOpened():
        ret, frame2 = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        newXs, newYs = estimateAllTranslation(startXs, startYs, frame1, frame2)
        Xs, Ys, newbbox = applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox)

        # Update feature points
        startXs, startYs = Xs, Ys
        n_features_left = np.sum(Xs != -1)
        print(f'# of Features: {n_features_left}')
        if n_features_left < 15:
            print('Generate New Features')
            startXs, startYs = getFeatures(cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY), newbbox)

        # Draw bounding boxes and features
        for j in range(n_object):
            (xmin, ymin, boxw, boxh) = cv2.boundingRect(newbbox[j, :, :].astype(int))
            frame2 = cv2.rectangle(frame2, (xmin, ymin), (xmin + boxw, ymin + boxh), (255, 0, 0), 2)
            for k in range(startXs.shape[0]):
                if startXs[k, j] != -1 and startYs[k, j] != -1:
                    frame2 = cv2.circle(frame2, (int(startXs[k, j]), int(startYs[k, j])), 3, (0, 0, 255), thickness=2)

        if play_realtime:
            cv2.imshow("Real-time Tracking", frame2)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop if 'q' is pressed
                break

        if save_to_file:
            out.write(frame2)

        # Update frame1 to the current frame2 for next iteration
        frame1 = frame2

    cap.release()
    if save_to_file:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    objectTracking(draw_bb=True, play_realtime=True, save_to_file=True)
'''
'''
import cv2
import numpy as np 

from getFeatures import getFeatures
from estimateAllTranslation import estimateAllTranslation
from applyGeometricTransformation import applyGeometricTransformation


def objectTracking(rawVideo, draw_bb=False, play_realtime=False, save_to_file=False):
    # initilize
    n_frame = 400
    frames = np.empty((n_frame,),dtype=np.ndarray)
    frames_draw = np.empty((n_frame,),dtype=np.ndarray)
    bboxs = np.empty((n_frame,),dtype=np.ndarray)
    for frame_idx in range(n_frame):
        _, frames[frame_idx] = rawVideo.read()

    # draw rectangle roi for target objects, or use default objects initilization
    if draw_bb:
        n_object = int(input("Number of objects to track:"))
        bboxs[0] = np.empty((n_object,4,2), dtype=float)
        for i in range(n_object):
            (xmin, ymin, boxw, boxh) = cv2.selectROI("Select Object %d"%(i),frames[0])
            cv2.destroyWindow("Select Object %d"%(i))
            bboxs[0][i,:,:] = np.array([[xmin,ymin],[xmin+boxw,ymin],[xmin,ymin+boxh],[xmin+boxw,ymin+boxh]]).astype(float)
    else:
        n_object = 1
        bboxs[0] = np.array([[[291,187],[405,187],[291,267],[405,267]]]).astype(float)
    
    if save_to_file:
        out = cv2.VideoWriter('output.avi',0,cv2.VideoWriter_fourcc('M','J','P','G'),20.0,(frames[i].shape[1],frames[i].shape[0]))
    
    # Start from the first frame, do optical flow for every two consecutive frames.
    startXs,startYs = getFeatures(cv2.cvtColor(frames[0],cv2.COLOR_RGB2GRAY),bboxs[0],use_shi=False)
    for i in range(1,n_frame):
        print('Processing Frame',i)
        newXs, newYs = estimateAllTranslation(startXs, startYs, frames[i-1], frames[i])
        Xs, Ys ,bboxs[i] = applyGeometricTransformation(startXs, startYs, newXs, newYs, bboxs[i-1])
        
        # update coordinates
        startXs = Xs
        startYs = Ys

        # update feature points as required
        n_features_left = np.sum(Xs!=-1)
        print('# of Features: %d'%n_features_left)
        if n_features_left < 15:
            print('Generate New Features')
            startXs,startYs = getFeatures(cv2.cvtColor(frames[i],cv2.COLOR_RGB2GRAY),bboxs[i])

        # draw bounding box and visualize feature point for each object
        frames_draw[i] = frames[i].copy()
        for j in range(n_object):
            (xmin, ymin, boxw, boxh) = cv2.boundingRect(bboxs[i][j,:,:].astype(int))
            frames_draw[i] = cv2.rectangle(frames_draw[i], (xmin,ymin), (xmin+boxw,ymin+boxh), (255,0,0), 2)
            for k in range(startXs.shape[0]):
                frames_draw[i] = cv2.circle(frames_draw[i], (int(startXs[k,j]),int(startYs[k,j])),3,(0,0,255),thickness=2)
        
        # imshow if to play the result in real time
        if play_realtime:
            cv2.imshow("win",frames_draw[i])
            cv2.waitKey(10)
        if save_to_file:
            out.write(frames_draw[i])
    
    if save_to_file:
        out.release()

    # loop the resulting video (for debugging purpose only)
    # while 1:
    #     for i in range(1,n_frame):
    #         cv2.imshow("win",frames_draw[i])
    #         cv2.waitKey(50)


if __name__ == "__main__":
    cap = cv2.VideoCapture("WhatsApp Video 2024-10-28 at 20.30.36.mp4")
    objectTracking(cap,draw_bb=True,play_realtime=True,save_to_file=True)
    cap.release()
'''