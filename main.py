from detect import detect_image
import torch
from torchvision import transforms
from PIL import Image
import time

device = 'cuda' if torch.cuda.is_available else 'cpu'
use_gpu = device == 'cuda'
# if use_gpu:
#     model = torch.load("weight/vectorize_model.pth")
# else:
vectorize_model = torch.load("weight/vectorize_model.pth", map_location=torch.device('cpu'))
transformer = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])


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


if __name__ == "__main__":
    image = Image.open("images/test.jpg")
    result_yolo = detect_image(weights="./weight/best.pt", source="./images",
                               data="./custom_data/yolo_deepfashion_data.yaml", conf_thres=0.1)

    respond_result = []
    for result in result_yolo:
        label = int(result['class'])
        xywh = result['xywh']
        x0, y0, x1, y1 = turn_xywh_to_coordination(image.size, xywh)
        crop_img = image.crop((x0, y0, x1, y1))
        vector = vectorize_image(image, vectorize_model, transformer)
        print(vector)
