import cv2
import numpy as np
import matplotlib.pyplot as plt


from util import get_parking_spots_bboxes, empty_or_not

def calc_diff(im1,  im2):
    return np.abs(np.mean(im1) - np.mean(im2)  )     



mask = "C:\\Users\\anixl\\projects\\parkedcar\\masks\\mask_1920_1080.png"
video_path= ' .\\parkedcar\\data\\parking_1920_1080_loop.mp4 '


mask = cv2.imread(mask, 0)
cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)
diffs = [None  for j in spots]

spots_status = [None for j in spots]
previous_frame = None

frame_nmr = 0
ret = True
step = 30
while ret:
    ret, frame = cap.read()

    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_idx, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1+h , x1:x1+h :]
            diffs[spot_idx] = calc_diff(spot_crop, previous_frame[y1:y1+h , x1:x1+h :])


    if frame_nmr % step == 0:
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j]/np.amax(diffs) > 0.4]
        for spot_idx in arr_:
            spot = spots[spot_idx]
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1+h , x1:x1+h :]
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_idx] = spot_status


    if frame_nmr % step == 0:
        previous_frame = frame.copy()

    for spot_idx, spot in enumerate(spots): 
        x1, y1, w, h = spots[spot_idx]
        spot_status = spots_status[spot_idx]
        if spot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
        else:
           frame = cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 0, 255), 2)

    cv2.rectangle( frame, (80, 28), (600, 80), (0, 0, 0), -1 )
    cv2.putText( frame, " Available Spots : {} / {} ".format(str(sum(spots_status)), str(len(spots_status))), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255,), 2  )
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame )
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    frame_nmr+=1
    

cap.release()
cv2.destroyAllWindows()
