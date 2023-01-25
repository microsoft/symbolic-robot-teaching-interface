import requests
import os
import cv2


def upload_data(upload_file_rgb, fs=30):
    url = 'URL'
    headers = {'accept': 'application/json'}
    files = {'upload_file': open(upload_file_rgb, 'rb')}
    data = {'fs': fs}
    response = requests.post(url, headers=headers,
                             files=files, data=data)
    #data = response.data()
    return response


def run(fp_out_mp4, scale=None, fs=30):
    # rescale video using opencv
    if scale is not None:
        cap = cv2.VideoCapture(fp_out_mp4)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w_new = int(w * scale)
        h_new = int(h * scale)
        fp_out_mp4_rescale = fp_out_mp4.replace('.mp4', '_rescale.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(fp_out_mp4_rescale, fourcc, fps, (w_new, h_new))
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (w_new, h_new))
                out.write(frame)
            else:
                break
        cap.release()
        out.release()
    else:
        fp_out_mp4_rescale = fp_out_mp4
    response = upload_data(fp_out_mp4_rescale, fs=fs)
    data = response.content
    # save the result
    path_output = fp_out_mp4_rescale.replace(".mp4", ".zip")
    with open(path_output, 'wb') as s:
        s.write(data)
    import shutil
    try:
        shutil.unpack_archive(path_output, path_output.replace('.zip', ''))
        # find the result file
        from glob import glob
        files = []
        for dir, _, _ in os.walk(
            os.path.join(
                path_output.replace(
                '.zip', ''))):
            files.extend(glob(os.path.join(dir, "segment.json")))
        return os.path.normpath(files[0])
    except BaseException:
        print('unpack failed')
        return None
