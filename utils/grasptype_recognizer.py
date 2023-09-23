import json
import requests
import os
import tempfile

def upload_data_image(fp_tmp_img, object_name="none"):
    url = 'http://localhost:8085/grasp_type_recognition'
    headers = {'accept': 'application/json'}
    with open(fp_tmp_img, 'rb') as f:
        files = {'upload_file': f}
        values = {'object_name': object_name}
        response = requests.post(url, headers=headers, files=files, data=values)
    return response

def upload_data_image_customvision(fp_tmp_img, object_name="none"):
    url = 'http://localhost:8085/grasp_type_recognition_customvision'
    headers = {'accept': 'application/json'}
    with open(fp_tmp_img, 'rb') as f:
        files = {'upload_file': f}
        values = {'object_name': object_name}
        response = requests.post(url, headers=headers, files=files, data=values)
    return response


def run_image(fp_tmp_img, object_name='none'):
    with tempfile.TemporaryDirectory() as output_dir_tmp:
        fp_tmp_out = os.path.join(output_dir_tmp, 'tmp.json')
        #response = upload_data_image(fp_tmp_img, object_name)
        response = upload_data_image_customvision(fp_tmp_img, object_name)
        data = response.content
        with open(fp_tmp_out, 'wb') as s:
            s.write(data)
        with open(fp_tmp_out) as json_file:
            data = json.load(json_file)
            return data['grasp_type']
