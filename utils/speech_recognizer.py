import json
import requests
import os

def upload_data(fp_audio_json, fp_segmentation):
    url = 'http://localhost:8082/audio_split_and_speech_recognition'
    headers = {'accept': 'application/json'}
    with open(fp_audio_json, 'rb') as audio_file, open(fp_segmentation, 'rb') as segmentation_file:
        files = {
            'upload_file': audio_file,
            'upload_json': segmentation_file
        }
        response = requests.post(url, headers=headers, files=files)

    return response


def run(fp_audio_json, fp_segmentation, output_dir):
    fp_tmp_zip = os.path.join(output_dir, 'speech_recognized.zip')
    response = upload_data(fp_audio_json, fp_segmentation)
    data = response.content
    with open(fp_tmp_zip, 'wb') as s:
        s.write(data)
    import shutil
    try:
        shutil.unpack_archive(fp_tmp_zip, fp_tmp_zip.replace('.zip', ''))
        from glob import glob
        files = []
        for dir, _, _ in os.walk(
            os.path.join(
                fp_tmp_zip.replace(
                '.zip', ''))):
            files.extend(glob(os.path.join(dir, "transcript.json")))
        return os.path.normpath(files[0])
    except BaseException:
        print('unpack failed')
