import PySimpleGUI as sg
import os
from pyk4a import Config, ImageFormat, PyK4A, PyK4ARecord
import cv2
from typing import Optional, Tuple
import numpy as np
import time
import pyaudio
import wave


def convert_to_bgra_if_required(color_format: ImageFormat, color_image):
    if color_format == ImageFormat.COLOR_MJPG:
        color_image = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
    elif color_format == ImageFormat.COLOR_NV12:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_NV12)
    elif color_format == ImageFormat.COLOR_YUY2:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_YUY2)
    return color_image


def colorize(
    image: np.ndarray,
    clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
    colormap: int = cv2.COLORMAP_HSV,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)
    return img


def get_file_info():
    layout = [
        [
            sg.Text(
                'Saving file name', size=(
                    40, 1), justification='center', font='Helvetica 20')], [
            sg.FolderBrowse(
                font='Helvetica 14'), sg.Text(
                "Folder name", font='Helvetica 14'), sg.InputText(font='Helvetica 14')], [
            sg.Text(
                "File name", font='Helvetica 14'), sg.InputText(font='Helvetica 14')], [
            sg.Submit(
                key="submit", font='Helvetica 14'), sg.Cancel(
                "Exit", font='Helvetica 14')]]

    window = sg.Window("file selection", layout, location=(400, 400))

    while True:
        event, values = window.read(timeout=100)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            fp_dir = None
            fp_base = None
            break
        elif event == 'submit':
            if values[0] == "" or values[1] == "":
                sg.popup("Enter file information")
                continue
            else:
                fp_dir = values[0]
                fp_base = values[1]
                break
    window.close()
    return fp_dir, fp_base


def ui(fp_dir, fp_base, extraction=False, size=(1280, 720), save_fps=0, save_raw=True, depth_debug=True):
    fp_mkv = None
    fp_out_mp4 = None
    fp_out_depth_npy = None
    fp_audio = None
    if extraction:
        color_frame_count = 0
        depth_frame_count = 0
        if save_fps > 0:
            fs = save_fps
        else:
            fs = 30
        W_save, H_save = size[0], size[1]
        depth_stack = []
        fp_out_mp4 = os.path.join(fp_dir, fp_base + ".mp4")
        fp_out_depth_npy = os.path.join(fp_dir, fp_base + "_depth.npy")
        videowriter = cv2.VideoWriter(
            fp_out_mp4, cv2.VideoWriter_fourcc(
                *"mp4v"), fs, (W_save, H_save))
        if depth_debug:
            fp_out_depth_mp4 = os.path.join(fp_dir, fp_base + "_depth.mp4")
            videowriter_d = cv2.VideoWriter(
                fp_out_depth_mp4, cv2.VideoWriter_fourcc(
                    *"mp4v"), fs, (W_save, H_save))

        # sg.theme('Black')
    deviceid = 0
    read_frames = 0
    imageformat = ImageFormat.COLOR_MJPG
    print(f"Starting device #{deviceid}")
    config = Config(color_format=imageformat)
    device = PyK4A(config=config, device_id=deviceid)
    device.start()

    if save_raw:
        fp_mkv = os.path.join(fp_dir, fp_base + ".mkv")
        print(f"Open record file {fp_mkv}")
        record = PyK4ARecord(device=device, config=config, path=fp_mkv)
        record.create()

    p = pyaudio.PyAudio()
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 512
    audio = pyaudio.PyAudio()
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    print("Select audio devive:")
    for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("ID " + str(i) + ": " + audio.get_device_info_by_host_api_device_index(0, i).get('name'))
    index = int(input())
    print("Recording via index "+str(index))
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, input_device_index=index,
                        frames_per_buffer=CHUNK)
    Recordframes = []
    recording_start_flag = False

    # Read frames from microphone and write to wav file
    fp_audio = os.path.join(fp_dir, fp_base + ".wav")
    # define the window layout
    layout = [
        [
            sg.Text(
                'Recorder', size=(
                    40, 1), justification='center', font='Helvetica 20')], [
            sg.Image(
                filename='', key='image')], [
            sg.Button(
                'Record', size=(
                    10, 1), font='Helvetica 14'), sg.Button(
                'Stop', size=(
                    10, 1), font='Helvetica 14'), sg.Button(
                'Exit', size=(
                    10, 1), font='Helvetica 14'), ]]
    # create the window and show it without the plot
    window = sg.Window('Recorder: Azure Kinect',
                       layout, location=(0, 0))
    recording = False
    # get current time in ms
    past_time = int(round(time.time() * 1000))
    past_time_save = past_time
    while True:
        event, values = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break

        elif event == 'Record':
            recording = True
            window['Record'].update(disabled=True)
        elif event == 'Stop':
            recording = False
            if save_raw:
                record.flush()
                record.close()
            break
        if recording:
            if not recording_start_flag:
                # get current available frames because these are not supposed to be included in the recording
                available_frames = stream.get_read_available()
                data = stream.read(available_frames)
                recording_start_flag = True

            capture = device.get_capture()
            if save_raw:
                record.write_capture(capture)

            if extraction:
                # get current time in ms
                current_time = int(round(time.time() * 1000))
                # calculate the time difference between current time and
                # past time
                time_diff = current_time - past_time_save
                # if time difference is greater than 1000 ms, we are
                # displaying frames per second
                # multiply 0.9 to adjust fps delay
                if time_diff > int(1000 * (1.0 / save_fps))*0.9:
                    past_time_save = current_time
                    if capture.color is not None:
                        frame = convert_to_bgra_if_required(
                            imageformat, capture.color)
                        resized = cv2.resize(
                            frame, (int(W_save), int(H_save)),
                            interpolation=cv2.INTER_NEAREST)
                        videowriter.write(resized)
                        color_frame_count += 1
                    if capture.transformed_depth is not None:
                        frame_d = capture.transformed_depth
                        resized_d = cv2.resize(
                            frame_d, (int(W_save), int(H_save)),
                            interpolation=cv2.INTER_NEAREST)
                        depth_stack.append(resized_d)
                        resized_d_color = colorize(resized_d, (None, 5000))
                        if depth_debug:
                            videowriter_d.write(resized_d_color)
                        depth_frame_count += 1

                    scale = 0.3
                    preview_image = resized
                    preview_image = cv2.resize(
                        resized, (int(W_save * scale), int(H_save * scale)),
                        interpolation=cv2.INTER_NEAREST)
                    cv2.putText(preview_image,
                                str("Preview: {0:.1f} FPS".format(
                                    1000.0 / time_diff)),
                                (15, 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (240, 230, 0),
                                1,
                                cv2.LINE_AA)
                    imgbytes = cv2.imencode('.png', preview_image)[
                        1].tobytes()  # ditto
                    window['image'].update(data=imgbytes)
            # audio
            available_frames = stream.get_read_available()
            data = stream.read(available_frames)
            Recordframes.append(data)
        else:  # preview mode
            current_time = int(round(time.time() * 1000))
            # calculate the time difference between current time and
            # past time
            time_diff = current_time - past_time
            capture = device.get_capture()
            if capture.color is not None:
                past_time = current_time
                frame = convert_to_bgra_if_required(
                    imageformat, capture.color)
                scale = 0.3
                W, H = frame.shape[1], frame.shape[0]
                resized = cv2.resize(
                    frame, (int(W * scale), int(H * scale)),
                    interpolation=cv2.INTER_NEAREST)
                cv2.putText(resized,
                            str("Preview: {0:.1f} FPS".format(
                                1000.0 / time_diff)),
                            (15, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (240, 230, 0),
                            1,
                            cv2.LINE_AA)
                imgbytes = cv2.imencode('.png', resized)[
                    1].tobytes()  # ditto
                window['image'].update(data=imgbytes)
    window.close()
    if save_raw:
        print(f"{record.captures_count} frames written.")
    else:
        print(f"{color_frame_count} frames written (RGB).")
        print(f"{depth_frame_count} frames written (D).")
    device.stop()
    stream.stop_stream()
    stream.close()
    p.terminate()
    wavefile = wave.open(fp_audio, 'wb')
    wavefile.setnchannels(CHANNELS)
    wavefile.setsampwidth(audio.get_sample_size(FORMAT))
    wavefile.setframerate(RATE)
    wavefile.writeframes(b''.join(Recordframes))
    wavefile.close()
    if extraction:
        videowriter.release()
        if depth_debug:
            videowriter_d.release()
        if len(depth_stack) > 0:
            depth_data = np.stack(depth_stack)
            np.save(fp_out_depth_npy, depth_data)
    ret = {"fp_mkv": fp_mkv,
           "fp_out_mp4": fp_out_mp4,
           "fp_out_depth_npy": fp_out_depth_npy,
           "fp_audio": fp_audio}
    return ret


def run(size=(1280, 720), save_fps=30, save_raw=True, extraction=False):
    fp_dir, fp_base = get_file_info()
    if fp_dir == None:
        return None, None, None, None
    ret = ui(fp_dir, fp_base, size=size, save_fps=save_fps,
             save_raw=save_raw, extraction=extraction)
    fp_mkv = ret["fp_mkv"]
    fp_audio = ret["fp_audio"]
    return fp_mkv, fp_audio, fp_dir, ret


if __name__ == '__main__':
    fp_dir, fp_base = get_file_info()
    ui(fp_dir, fp_base, size=(640, 360),
       save_fps=5, save_raw=True, extraction=True)
