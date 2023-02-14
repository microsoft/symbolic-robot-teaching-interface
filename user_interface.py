
import numpy as np
import PySimpleGUI as sg
import cv2
from pathlib import Path
import json
import os
import pickle
import glob
import open3d as o3d

import utils.video_segmentator as video_segmentator
import utils.task_recognizer as task_recognizer
import utils.task_compiler as task_compiler
import utils.speech_recognizer as speech_recognizer
import utils.recorder_gui as recorder_gui
import utils.preview_gui as preview_gui
import utils.speech_synthesizer as speech_synthesizer
import asyncio
import shutil
speech_synthesizer_azure = speech_synthesizer.speech_synthesizer(speech_synthesis_voice_name="en-US-TonyNeural")

def compile_task(
        task,
        verbal_input,
        fp_mp4,
        fp_depth_npy,
        segment_timings_frame,
        output_dir_daemon,
        hand_laterality):
    daemon = task_compiler.task_daemon(
        task,
        verbal_input,
        fp_mp4,
        fp_depth_npy,
        segment_timings_frame,
        output_dir_daemon,
        hand_laterality=hand_laterality)
    daemon.set_skillparameters()
    daemon.dump_json()
    return daemon


async def run_daemon(loop,
                     task_list, verbal_input_list, fp_mp4, fp_depth_npy,
                     segment_timings_frame_list, output_dir_daemon, hand_laterality):
    sem = asyncio.Semaphore(5)

    async def run_request(task, verbal_input, fp_mp4,
                          fp_depth_npy, segment_timings_frame, output_dir_daemon, hand_laterality):
        async with sem:
            return await loop.run_in_executor(None, compile_task,
                                              task,
                                              verbal_input,
                                              fp_mp4,
                                              fp_depth_npy,
                                              segment_timings_frame,
                                              output_dir_daemon,
                                              hand_laterality)
    damon_list = [
        run_request(
            task_list[i],
            verbal_input_list[i],
            fp_mp4,
            fp_depth_npy,
            segment_timings_frame_list[i],
            output_dir_daemon,
            hand_laterality) for i in range(
            len(task_list))]
    return await asyncio.gather(*damon_list)


def file_read():
    layout = [
        [
            sg.Submit("Record", font='Helvetica 14')
        ],
        [
            sg.FileBrowse(key="mp4file", font='Helvetica 14'),
            sg.Text("mp4 file", font='Helvetica 14'),
            sg.InputText(font='Helvetica 14')
        ],
        [
            sg.FileBrowse(key="depthfile", font='Helvetica 14'),
            sg.Text("depth file", font='Helvetica 14'),
            sg.InputText(font='Helvetica 14')
        ],
        [
            sg.FileBrowse(key="audiofile", font='Helvetica 14'),
            sg.Text("audio file (if any)", font='Helvetica 14'),
            sg.InputText(font='Helvetica 14')
        ],
        [sg.Submit(key="submit", font='Helvetica 14'), sg.Cancel("Exit", font='Helvetica 14')]
    ]

    window = sg.Window("file selection", layout)

    while True:
        event, values = window.read(timeout=100)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            exit(0)
        elif event == 'Record':
            # disable this window
            window.close()
            fp_mkv, fp_audio, fp_dir, ret = recorder_gui.run(size=(1280,720), save_fps=5, save_raw=False, extraction=True)
            if fp_dir == None:
                exit(0)
            assert fp_mkv == None
            fp_mp4 = ret["fp_out_mp4"]
            fp_depth = ret["fp_out_depth_npy"]
            window.close()
            return (Path(fp_mp4),Path(fp_depth)), Path(fp_audio), fp_dir
        elif event == 'submit':
            if values[0] == "":
                sg.popup("no video file input")
                event = ""
            else:
                fp_mp4 = values[0]
                fp_depth = values[1]
                fp_audio = values[2]
                if values[1] == "":
                    window.close()
                    return (Path(fp_mp4),Path(fp_depth)), None, None
                break
        fp_mp4 = values[0]
        if len(fp_mp4)>0:
            window[1].update(fp_mp4.split('.')[0]+"_depth.npy")
            window[2].update(fp_mp4.split('.')[0]+".wav")
    window.close()
    return (Path(fp_mp4),Path(fp_depth)), Path(fp_audio), None

if __name__ == '__main__':
    debug = False
    # get file paths
    fp_source, fp_audio, output_dir_name_root = file_read()
    fp_mp4_source,fp_depth_source = fp_source[0],fp_source[1]
    if output_dir_name_root is None:
        output_dir_name_root = os.path.dirname(fp_mp4_source)
    filename_mp4 = os.path.basename(fp_mp4_source)
    output_dir = os.path.join(
        os.getcwd(),
        output_dir_name_root,
        filename_mp4.split('.')[0],
        'task_recognition')
    output_dir_daemon = os.path.join(
        os.getcwd(),
        output_dir_name_root,
        filename_mp4.split('.')[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        debug = False

    print('copying mp4 and depth..')
    fp_mp4 = os.path.join(output_dir, filename_mp4)
    fp_depth_npy = os.path.join(
        output_dir,
        filename_mp4.split('.')[0] +
        '_depth.npy')
    if not os.path.exists(fp_mp4) or not os.path.exists(fp_depth_npy):
        shutil.copyfile(fp_mp4_source,fp_mp4)
        shutil.copyfile(fp_depth_source,fp_depth_npy)
        debug = False
    print('done')

    # segment video
    print('segmenting video...')
    speech_synthesizer_azure.synthesize_speech('Splitting the video into segments.')
    if debug:
        # find a file path to 'segment.json'
        fp_segmentation = os.path.join(output_dir,
                                       filename_mp4.split('.')[0]+'_rescale',
                                       filename_mp4.split('.')[0]+'_rescale',
                                       'segment.json')
        if not os.path.exists(fp_segmentation):
            debug = False
    if not debug:
        cap = cv2.VideoCapture(fp_mp4)
        fps_input = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        fp_segmentation = video_segmentator.run(fp_mp4, scale=0.3, fs=fps_input)
    if os.path.exists(fp_segmentation):
        with open(fp_segmentation, 'r') as f:
            segment_data = json.load(f)
            segment_timings = segment_data['frame_minimum']
            segment_timings_sec = segment_data['time_minimum']
    timeparts_frame = []
    timeparts_sec = []
    for i in range(len(segment_timings) - 1):
        timeparts_frame.append(
            (segment_timings[i], segment_timings[i + 1]))
        timeparts_sec.append(
            (segment_timings_sec[i], segment_timings_sec[i + 1]))
    print('done')

    # split videos based on timepart
    print('splitting videos...')
    if debug:
        fp_splitvideo = glob.glob(os.path.join(output_dir, 'segment_part*'))
        if len(fp_splitvideo) == 0:
            debug = False
    if not debug:
        cap = cv2.VideoCapture(fp_mp4)
        ret, frame = cap.read()
        h, w, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fp_splitvideo = [
            os.path.join(
                output_dir,
                f"segment_part_{start}-{end}.mp4") for start,
            end in timeparts_frame]
        writers = [cv2.VideoWriter(fp_tmp, fourcc, 5.0, (w, h))
                   for fp_tmp in fp_splitvideo]
        f = 0
        while ret:
            f += 1
            for i, part in enumerate(timeparts_frame):
                start, end = part
                if start <= f <= end:
                    writers[i].write(frame)
            ret, frame = cap.read()
        for writer in writers:
            writer.release()
        cap.release()
    print('done')

    # analyze audio file
    print('analyzing audios...')
    speech_synthesizer_azure.synthesize_speech('Analyzing speech.')
    audio_data = {}
    transcript = []

    if fp_audio is not None:
        if debug:
            fp_text = os.path.join(
                output_dir,
                'speech_recognized',
                'transcript.json')
            if not os.path.exists(fp_text):
                debug = False
        if not debug:
            fp_text = speech_recognizer.run(
                fp_audio, fp_segmentation, output_dir)
        if os.path.exists(fp_text):
            with open(fp_text, 'r') as f:
                audio_data = json.load(f)
                transcript = audio_data['recognized_text']
    print('done')

    # confirm the verbal input
    print('confirming the verbal input...')
    speech_synthesizer_azure.synthesize_speech('Please confirm the verbal input.')
    transcript_confirmed = []
    if debug:
        fp_text_confirmed = os.path.join(
            output_dir, 'transcript_confirmed.json')
        if not os.path.exists(fp_text_confirmed):
            debug = False
    if not debug:
        for i, fp_video_item in enumerate(fp_splitvideo):
            transcript_item = ""
            if i < len(transcript):
                transcript_item = transcript[i]
            confirmed_transcript_item = preview_gui.preview_gui(
                fp_video_item, transcript_item).run()
            transcript_confirmed.append(confirmed_transcript_item)
        dump = {}
        dump['recognized_text'] = transcript_confirmed
        dump['fp_video'] = fp_splitvideo
        dump['segment_timings_frame'] = timeparts_frame
        dump['segment_timings_sec'] = timeparts_sec
        fp_text_confirmed = os.path.join(
            output_dir, "transcript_confirmed.json")
        with open(fp_text_confirmed, 'w') as f:
            json.dump(dump, f, indent=4)

    with open(fp_text_confirmed, 'r') as f:
        data = json.load(f)
        transcript_confirmed = data['recognized_text']
        fp_video = data['fp_video']
        timeparts_frame = data['segment_timings_frame']
        timeparts_sec = data['segment_timings_sec']
    print('done')

    # Write the result of segmentation after user's confirmation
    print('writing video...')
    if not debug:
        parts_confirmed = []
        parts_confirmed_show = []
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        for i, part in enumerate(transcript_confirmed):
            if part != "":
                parts_confirmed.append(timeparts_frame[i])
                parts_confirmed_show.append(part)
        cap = cv2.VideoCapture(fp_mp4)
        ret, frame = cap.read()
        h, w, _ = frame.shape
        writer = cv2.VideoWriter(
            os.path.join(
                output_dir, 'segment_result.mp4'), fourcc, 5.0, (w, h))
        f = 0
        while ret:
            f += 1
            currentseg = None
            for i, part in enumerate(parts_confirmed):
                start, end = part
                if start <= f <= end:
                    currentseg = i
            # draw currentseg info to the frame
            if currentseg is not None:
                # font size = 5
                cv2.putText(
                    frame,
                    f"Segment index: {currentseg}",
                    (10,
                        60),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    4,
                    color=(
                        30,
                        30,
                        255),
                    thickness=4)
            else:
                cv2.putText(
                    frame,
                    f"Not manipulation",
                    (10,
                        60),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    4,
                    color=(
                        30,
                        30,
                        255),
                    thickness=4)
            writer.write(frame)
            ret, frame = cap.read()
        writer.release()
        cap.release()
    print('done')

    print('encoding task models...')
    speech_synthesizer_azure.synthesize_speech('Encoding task models.')
    if debug:
        fp_daemon = os.path.join(output_dir, "daemons.pkl")
        if not os.path.exists(fp_daemon):
            debug = False
    if not debug:
        #  recognize tasks
        print('recognizing tasks...')
        fp_task_recognized = task_recognizer.run_luisbased(
            fp_text_confirmed, output_dir)
        print('done')

        print('Checking hand laterality...')
        hand_laterality = ''
        with open(fp_task_recognized, 'r') as f:
            task_recognized = json.load(f)
        for i, task in enumerate(task_recognized["recognized_tasks"]):
            segment_timings_frame = task_recognized["segment_timings_frame"][i]
            verbal_input = task_recognized["recognized_text"][i]
            if task == "GRASP":
                print("encoding: " + task)
                daemon = task_compiler.task_daemon(
                    task,
                    verbal_input,
                    fp_mp4,
                    fp_depth_npy,
                    segment_timings_frame,
                    output_dir_daemon,
                    hand_laterality='unknown')
                hand_laterality = daemon.hand_laterality
        #hand_laterality = 'right'
        print(hand_laterality)

        print('compiling task models...')
        daemons = []
        manipulation_flag = False
        with open(fp_task_recognized, 'r') as f:
            task_recognized = json.load(f)

        task_list = []
        verbal_input_list = []
        segment_timings_frame_list = []
        for i, task in enumerate(task_recognized["recognized_tasks"]):
            segment_timings_frame = task_recognized["segment_timings_frame"][i]
            verbal_input = task_recognized["recognized_text"][i]
            if task == "NOTTASK" or task == "UNKNOWN":
                continue
            else:
                print("encoding: " + task)
                print(segment_timings_frame)
                task_list.append(task)
                verbal_input_list.append(verbal_input)
                segment_timings_frame_list.append(segment_timings_frame)
        loop = asyncio.get_event_loop()

        valid_task_list_i = []
        manipulation_flag = False
        for i, item in enumerate(task_list):
            if manipulation_flag == False and item == "GRASP":
                manipulation_flag = True
                if i > 0 and task_list[i - 1] == "PTG12":
                    valid_task_list_i.append(i - 1)
            if manipulation_flag:
                valid_task_list_i.append(i)

        manipulation_flag = False
        valid_task_list_i.sort()
        task_list = [task_list[i] for i in valid_task_list_i]
        verbal_input_list = [verbal_input_list[i] for i in valid_task_list_i]
        segment_timings_frame_list = [
            segment_timings_frame_list[i] for i in valid_task_list_i]

        daemons = loop.run_until_complete(
            run_daemon(
                loop,
                task_list,
                verbal_input_list,
                fp_mp4,
                fp_depth_npy,
                segment_timings_frame_list,
                output_dir_daemon,
                hand_laterality))

        fp_daemon = os.path.join(output_dir, "daemons.pkl")
        with open(fp_daemon, 'wb') as f:
            pickle.dump(daemons, f)
    print('done')

    fp_daemon = os.path.join(output_dir, "daemons.pkl")
    with open(fp_daemon, 'rb') as f:
        daemons = pickle.load(f)

    # concatenate the task sequence
    task_models = []
    for daemon in daemons:
        task_models.append(daemon.taskmodel_json)

    task_models_save = []
    # manually modify the task sequence (parameter filling)
    for i, item in enumerate(task_models):
        if i > 0 and item["_task"] == "GRASP":
            item_pre = task_models[i - 1]
            # task_models_save.append(item_pre)
            item["prepre_grasp_position"]["value"] = item_pre["start_position"]["value"]

    for item in task_models:
        task_models_save.append(item)

    task_models_save.append({"_task": "END"})

    # save the task sequence
    task_models_save_json = {}
    task_models_save_json["version"] = "1.0"
    task_models_save_json["rawdata_path"] = str(fp_mp4_source)
    task_models_save_json["task_models"] = task_models_save
    fp_task_sequence = os.path.join(output_dir, "task_models.json")
    with open(fp_task_sequence, 'w') as f:
        json.dump(task_models_save_json, f, indent=4)

    # concatenate the hand points
    if len(daemons) > 0:
        pcd_list = []
        if daemons[0].fp_3dmodel_hand_first is not None:
            pcd = o3d.io.read_point_cloud(daemons[0].fp_3dmodel_hand_first)
            pcd.remove_non_finite_points()
            pcd.uniform_down_sample(7)
            # transform the coordinates to the original image
            if daemons[0].rot_mat_4x4_marker_to_camera is not None:
                pcd.transform(daemons[0].rot_mat_4x4_marker_to_camera)
            pcd_list.append(pcd)
        for daemon in daemons:
            if daemon.fp_3dmodel_hand_last is not None:
                pcd = o3d.io.read_point_cloud(daemon.fp_3dmodel_hand_last)
                pcd.remove_non_finite_points()
                pcd.uniform_down_sample(7)
                # transform the coordinates to the original image
                if daemon.rot_mat_4x4_marker_to_camera is not None:
                    pcd.transform(daemon.rot_mat_4x4_marker_to_camera)
                pcd_list.append(pcd)
            if daemon.task == "GRASP":
                if daemon.fp_3dmodel_object_first is not None:
                    pcd = o3d.io.read_point_cloud(
                        daemon.fp_3dmodel_object_first)
                    pcd.remove_non_finite_points()
                    pcd.uniform_down_sample(7)
                    # transform the coordinates to the original image
                    if daemon.rot_mat_4x4_marker_to_camera is not None:
                        pcd.transform(daemon.rot_mat_4x4_marker_to_camera)
                    pcd_list.append(pcd)
                    pcd_list.append(pcd)
            if daemon.task == "PTG5":
                hand_position = daemon.taskmodel_json["hand_trajectory"]["value"]
                for point in hand_position:
                    mesh = o3d.geometry.TriangleMesh.create_sphere(
                        radius=0.005)
                    mesh.translate(tuple(point))
                    mesh.paint_uniform_color([1, 0, 0])
                    pcd_list.append(mesh)
                hand_position = np.array(hand_position)
                xi = hand_position[:, 0]
                yi = hand_position[:, 1]
                rotation_center_position = daemon.taskmodel_json["rotation_center_position"]["value"]
                a = rotation_center_position[0]
                b = rotation_center_position[1]
                r = daemon.taskmodel_json["rotation_radius"]["value"]
                import utils.trajection_fitting as trajection_fitting
                xi, yi = trajection_fitting.generate_circular_points(a, b, r, np.pi / 20.0)
                for x, y in zip(xi, yi):
                    mesh = o3d.geometry.TriangleMesh.create_sphere(
                        radius=0.005)
                    mesh.translate((x, y, rotation_center_position[2]))
                    mesh.paint_uniform_color([0, 1, 0])
                    pcd_list.append(mesh)
        o3d.visualization.draw_geometries(pcd_list)

    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient
    fp_task_sequence = os.path.join(output_dir, "task_models.json")
    account_url = "BLOB_URL"
    print('uploading to blob storage')
    default_credential = DefaultAzureCredential(exclude_environment_credentials=True, exclude_shared_token_cache_credential=True)
    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient(account_url, credential=default_credential)
    container_name = 'storage'
    blob_client = blob_service_client.get_blob_client(container=container_name, blob='latest.json')
    with open(file=fp_task_sequence, mode="rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    print('done!!!')
    speech_synthesizer_azure.synthesize_speech('Compile done.')
