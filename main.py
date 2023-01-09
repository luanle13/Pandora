from detect import detect_image

if __name__ == "__main__":
    print("okokok")
    result = detect_image(weights="./weight/best.pt", source="./images", data="./custom_data/yolo_deepfashion_data.yaml", conf_thres=0.1)
    print(result)
