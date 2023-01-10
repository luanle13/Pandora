from detect import detect_image
import torch
from torchvision import transforms
import pandas as pd
from PIL import Image
import numpy as np
from scipy import spatial
from flask import Flask, flash, request, redirect, url_for
import base64


def vectorize_image(img, model, transformer):
    img_transformed = transformer(img).float()
    img_transformed = img_transformed.unsqueeze_(0)
    img_transformed = img_transformed.to("cpu")
    with torch.no_grad():
        model.eval()
        output = model(img_transformed)
    return output.squeeze().cpu().detach().numpy()


def turn_xywh_to_coordination(image_size, xywh):
    w, h = image_size
    x_bbox = xywh[0] * w
    w_bbox = xywh[2] * w
    y_bbox = xywh[1] * h
    h_bbox = xywh[3] * h
    x0 = x_bbox - w_bbox / 2
    y0 = y_bbox - h_bbox / 2
    x1 = x_bbox + w_bbox / 2
    y1 = y_bbox + w_bbox / 2
    return x0, y0, x1, y1


def get_top(top, vector, label):
    return_array = []
    list_cos_dist = []
    candidate_indexes = []
    i = 0
    for index, row in database.iterrows():
        if int(row['category_num']) == label:
            vec = vectorize_db.iloc[index, 1:].to_numpy()
            cosine_distance = spatial.distance.cosine(vector, vec)
            list_cos_dist.append(cosine_distance)
            candidate_indexes.append(index)
    ranked_dist_indexes = np.argsort(list_cos_dist)
    for index in range(0, top):
        id = candidate_indexes[ranked_dist_indexes[index]]
        return_array.append({'dir_img': database['dir_img'][id], 'label': database['category'][id]})
    return return_array


def createDicts():
    dict_cate_num = {}
    dict_num_cate = {}
    with open('category.txt') as f:
        i = 0
        while True:
            category = f.readline().strip()
            if not category:
                break
            dict_cate_num[category] = i
            dict_num_cate[i] = category
            i += 1
    return dict_cate_num, dict_num_cate


app = Flask(__name__)
device = 'cuda' if torch.cuda.is_available else 'cpu'
use_gpu = device == 'cuda'
database = pd.read_csv("database/db_data.csv")
vectorize_db = pd.read_csv("database/vectorized.csv")
vectorize_model = torch.load("weight/vectorize_model.pth", map_location=torch.device('cpu'))
transformer = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
dict_cate_num, dict_num_cate = createDicts()


@app.route('/', methods=['POST'])
def get_top_similar_images():
    body = request.json
    if 'image' not in body:
        flash('No image found')
        return redirect(request.url)
    elif not body['image']:
        flash('Image can not be null')
    else:
        image_data = base64.b64decode(body['image'])
        with open("images/test.jpg", 'wb') as f:
            f.write(image_data)

        result_yolo = detect_image(weights="./weight/best.pt", source="./images",
                                   data="./custom_data/yolo_deepfashion_data.yaml", conf_thres=0.2)
        image = Image.open("images/test.jpg")
        respond_result = {}
        for index, result in enumerate(result_yolo):
            label = int(result['class'])
            xywh = result['xywh']
            x0, y0, x1, y1 = turn_xywh_to_coordination(image.size, xywh)
            crop_img = image.crop((x0, y0, x1, y1))
            vector = vectorize_image(crop_img, vectorize_model, transformer)
            top_similar = get_top(3, vector, label)
            list_b64_string = []
            for obj in top_similar:
                with open(f"database/images/{obj['dir_img']}", 'rb') as f:
                    b64_str = base64.b64encode(f.read())
                    list_b64_string.append(str(b64_str.decode('utf-8')))
            respond_result[index] = {'category': dict_num_cate[label], 'list_b64_str': list_b64_string}
        return respond_result

if __name__ == "__main__":
    app.run()