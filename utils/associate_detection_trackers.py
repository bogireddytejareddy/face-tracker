import numpy as np
from utils.iou import iou
from sklearn.utils.linear_assignment_ import linear_assignment

def associate_detections_to_trackers(detections, trackers, iou_threshold = 0.3):
    """(numpy.array, numpy.array, int) -> numpy.array, numpy.array, numpy.array

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """

    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,4),dtype=int)

    iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

    for d, det in enumerate(detections):
        for t,trk in enumerate(trackers):
            iou_matrix[d,t] = iou(det,trk)

    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:, 0]):
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0],m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))

    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
