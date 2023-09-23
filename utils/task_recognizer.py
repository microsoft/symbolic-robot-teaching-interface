import json
import requests
import os
import tempfile

def upload_data_modelbased(upload_json):
    url = 'http://localhost:8082/text_based_taskrecognition_modelbased'
    headers = {'accept': 'application/json'}
    with open(upload_json, 'rb') as f:
        data = {'upload_json': f}
        response = requests.post(url, headers=headers, files=data)
    return response

def upload_data_luisbased(upload_json):
    url = 'http://localhost:8082/text_based_taskrecognition_luisbased'
    headers = {'accept': 'application/json'}
    with open(upload_json, 'rb') as f:
        data = {'upload_json': f}
        response = requests.post(url, headers=headers, files=data)
    return response

def upload_data(upload_json):
    url = 'http://localhost:8082/text_based_taskrecognition'
    headers = {'accept': 'application/json'}
    with open(upload_json, 'rb') as f:
        data = {'upload_json': f}
        response = requests.post(url, headers=headers, files=data)
    return response

def upload_data_object_recognition(upload_json):
    url = 'http://localhost:8082/text_based_objectrecognition'
    headers = {'accept': 'application/json'}
    with open(upload_json, 'rb') as f:
        data = {'upload_json': f}
        response = requests.post(url, headers=headers, files=data)
    return response


def run(fp_task_json, output_dir):
    fp_tmp_json = os.path.join(output_dir, 'task_recognized.json')
    response = upload_data(fp_task_json)
    data = response.content
    with open(fp_tmp_json, 'wb') as s:
        s.write(data)
    return fp_tmp_json


def run_modelbased(fp_task_json, output_dir):
    fp_tmp_json = os.path.join(output_dir, 'task_recognized.json')
    response = upload_data_modelbased(fp_task_json)
    data = response.content
    with open(fp_tmp_json, 'wb') as s:
        s.write(data)
    # open the json file
    with open(fp_tmp_json, 'r') as f:
        data = json.load(f)
    # save the json
    with open(fp_tmp_json, 'w') as f:
        json.dump(data, f, indent=4)
    return fp_tmp_json


def run_luisbased(fp_task_json, output_dir):
    fp_tmp_json = os.path.join(output_dir, 'task_recognized.json')
    #response = upload_data_modelbased(fp_task_json)
    response = upload_data_luisbased(fp_task_json)
    data = response.content
    with open(fp_tmp_json, 'wb') as s:
        s.write(data)
    # open the json file
    with open(fp_tmp_json, 'r') as f:
        data = json.load(f)
    # save the json
    with open(fp_tmp_json, 'w') as f:
        json.dump(data, f, indent=4)
    return fp_tmp_json

def run_object_recognition(textual_input):
    with tempfile.TemporaryDirectory() as output_dir_tmp:
        fp_tmp_json = os.path.join(output_dir_tmp, 'tmp.json')
        data = {}
        data['recognized_text'] = [textual_input]
        with open(fp_tmp_json, 'w') as s:
            json.dump(data, s)
        response = upload_data_object_recognition(fp_tmp_json)
        data = response.content
        # encode binary data to json
        data = json.loads(data)
        return data['recognized_objects'][0], data['recognized_attributes'][0]


if __name__ == '__main__':
    response = upload_data_modelbased('grasp the cup')
    data = response.content
    # encode binary data to json
    data = json.loads(data)
    print(data['1'].split(':')[0])
