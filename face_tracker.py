import cv2
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from core.facedetector import FaceDetector
from utils.associate_detection_trackers import associate_detections_to_trackers
from filterpy.kalman import KalmanFilter

class KalmanTracker(object):
    counter = 1
    def __init__(self, dets):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = np.array([dets[0], dets[1], dets[2], dets[3]]).reshape((4, 1))
        self.id = KalmanTracker.counter
        KalmanTracker.counter += 1
    
    def __call__(self):
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        return self.kf.x
    
    def correction(self, measurement):
        self.kf.update(measurement)

    def get_current_x(self):
        bbox = (np.array([self.kf.x[0], self.kf.x[1], self.kf.x[2], self.kf.x[3]]).reshape((1, 4)))
        return bbox

class FaceTracker(object):
    def __init__(self):
        self.current_trackers = []

    def __call__(self, detections):
        retain_trackers = []

        if len(self.current_trackers) == 0:
            self.current_trackers = []
            for d in range(len(detections)):
                tracker = KalmanTracker(detections[d, :-1])
                measurement = np.array((4,1), np.float32)
                measurement = np.array([[int(detections[d, 0])], [int(detections[d, 1])], [int(detections[d, 2])],
                                        [int(detections[d, 3])]], np.float32)
                tracker.correction(measurement)
                self.current_trackers.append(tracker)
            
            for trk in self.current_trackers:
                d = trk.get_current_x()
                retain_trackers.append(np.concatenate((d[0], [trk.id])).reshape(1,-1))

            if(len(retain_trackers) > 0):
                return np.concatenate(retain_trackers)
        
            return np.empty((0,5))
        
        else:
            predicted_trackers = []
            for t in range(len(self.current_trackers)):
                predictions = self.current_trackers[t]()[:4]
                predicted_trackers.append(predictions)

            predicted_trackers = np.asarray(predicted_trackers)

            matched, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(detections[:, :-1], 
                                                                                                predicted_trackers)
            
            print ('Matched Detections & Trackers', len(matched))
            print ('Unmatched Detections', len(unmatched_detections))
            print ('Unmatched Trackers', len(unmatched_trackers))
            print ('Current Trackers', len(self.current_trackers))

            for t in range(len(self.current_trackers)):
                if(t not in unmatched_trackers):
                    d = matched[np.where(matched[:,1]==t)[0], 0]
                    self.current_trackers[t].correction(np.array([detections[d, 0], detections[d, 1], 
                    detections[d, 2], detections[d, 3]]).reshape((4, 1)))

            for i in unmatched_detections:
                tracker = KalmanTracker(detections[i, :-1])
                measurement = np.array((4,1), np.float32)
                measurement = np.array([[int(detections[i, 0])], [int(detections[i, 1])], [int(detections[i, 2])],
                                        [int(detections[i, 3])]], np.float32)
                tracker.correction(measurement)
                self.current_trackers.append(tracker)

            for index in sorted(unmatched_trackers, reverse=True):
                del self.current_trackers[index]
            
            for trk in self.current_trackers:
                d = trk.get_current_x()
                retain_trackers.append(np.concatenate((d[0], [trk.id])).reshape(1,-1))

        if(len(retain_trackers) > 0):
            return np.concatenate(retain_trackers)

        return np.empty((0,5))

def read_detect_track_faces(videopath, facedetector, display=True):
    facetracker = FaceTracker()
    detection_frame_rate = 5

    jsonfile = open('./meta/' + videopath.split('/')[2].split('.')[0] + '_tracking_meta.json', 'w')

    videocapture = cv2.VideoCapture(videopath)
    success, frame = videocapture.read()
    frame_number = 1

    if display:
        colours = np.random.rand(32,3)
        plt.ion()
        fig = plt.figure()

    while success:
        success, frame = videocapture.read()

        if display:
            ax1 = fig.add_subplot(111, aspect='equal')
            ax1.imshow(frame)
            plt.title('Tracked Targets')

        if (frame_number % detection_frame_rate == 0) or (frame_number == 1):
            faces, _ = facedetector.detect(frame)

        trackers = facetracker(faces)
        frame_number += 1

        for tracker in trackers:
            data = {'frame number' : str(frame_number + 1), 'person number' : str(int(tracker[-1])),
                    'bounding box' : str(tracker[:-1])}
            json.dump(data, jsonfile)
            jsonfile.write('\n')

            if display:
                tracker = tracker.astype(np.int32)
                ax1.add_patch(patches.Rectangle((tracker[0], tracker[1]), tracker[2] - tracker[0], tracker[3] - tracker[1],
                fill=False, lw=3, ec=colours[tracker[4]%32, :]))
                plt.text(tracker[2] , tracker[3], 'Person ' + str(tracker[4]), fontsize=5, color=colours[tracker[4]%32, :])
        
        if(display):
          fig.canvas.flush_events()
          plt.draw()
          ax1.cla()

def parse_args():
    parser = argparse.ArgumentParser(description='Tracking Arguments')
    parser.add_argument('--detector', default='RetinaFaceDetector', help='Face Detector')
    parser.add_argument('--gpu', default=-1, help='CUDA Availability')
    parser.add_argument('--videopath', help='Input Video Path')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    detector_name = args.detector
    detector_params = {'gpu' : args.gpu}
    videopath = args.videopath

    facedetector = FaceDetector(detector_name, detector_params)

    read_detect_track_faces(videopath, facedetector)
