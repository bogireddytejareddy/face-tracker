import numpy as np
from insightface.RetinaFace.retinaface import RetinaFace

class RetinaFaceDetector(object):
    def __init__(self, detector_params):
        self.detector = RetinaFace('./insightface/RetinaFace/model/R50', 0, detector_params['gpu'], 'net3')

    def __call__(self, img):
        thresh = 0.8
        scales = [1024, 1980]

        im_shape = img.shape
        target_size = scales[0]
        max_size = scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)

        scales = [im_scale]
        flip = False

        faces, landmarks = self.detector.detect(img, thresh, scales=scales, do_flip=flip)
        return faces, landmarks

class FaceDetector(object):
    def __init__(self, detectorname, detector_params):
        if detectorname == 'RetinaFaceDetector':
            self.detector = RetinaFaceDetector(detector_params)
    
    def detect(self, image):
        return self.detector(image)