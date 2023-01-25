from logging import captureWarnings
import cv2
import os
import numpy as np
rootdir = 'DIR'
fp_mp4 = 'file_MP4'
latel = 'right'
# list of files in directory
file_list = os.listdir(rootdir)
fp_focus = [i for i in file_list if i.startswith(
    latel) and not i.endswith('_overlay.jpg')]
print(fp_focus)
num = [int(i.split('.')[0].split('_')[1]) for i in fp_focus]
print(np.sort(num))
# get file size
fp_frame = rootdir + '/' + latel + '_' + str(num[0]) + '.jpg'
frame = cv2.imread(fp_frame)
h, w, _ = frame.shape

fp_video_out = rootdir + '/' + 'video_out.mp4'
fps = 10
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(fp_video_out, fourcc, fps, (w, h))
for i in np.sort(num):
    fp_frame = rootdir + '/' + latel + '_' + str(i) + '.jpg'
    frame = cv2.imread(fp_frame)
    out.write(frame)
out.release()
print('Done')


# open the file and save of frames in num
cap = cv2.VideoCapture(fp_mp4)
ret, frame = cap.read()
h, w, _ = frame.shape
fp_video_out = rootdir + '/' + 'video_out_corresponding.mp4'
fps = 10
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(fp_video_out, fourcc, fps, (w, h))
i = 0
while (ret):
    if i in num:
        out.write(frame)
    ret, frame = cap.read()
    i += 1
cap.release()
out.release()
print('Done')
