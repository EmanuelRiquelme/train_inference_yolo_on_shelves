from ultralytics import YOLO

def save_model_info(model_name, model_data, mAP, mAP_50, mAP_75):
    with open(f'{model_name}_val_metrics.txt', 'w') as file:
        file.write(f"Model Name: {model_name}\n")
        file.write(f"Model Data: {model_data}\n")
        file.write(f"mAP: {mAP}\n")
        file.write(f"mAP_50: {mAP_50}\n")
        file.write(f"mAP_75: {mAP_75}\n")

def val(model_name,model_data):
    model = YOLO(model_name)
    metrics = model.val(data=model_data,
                                   imgsz=640,
                                   batch=12
                                   )
    mAP=metrics.box.map    
    mAP_50 = metrics.box.map50  
    mAP_75 = metrics.box.map75 
    save_model_info(model_name, model_data, mAP, mAP_50, mAP_75)
    print(f'mAP: {mAP}\n mAP_50: {mAP_50}\n mAP_75: {mAP_75}')
    print(f'file saved at: {model_name}_val_metrics.txt')

if __name__ == '__main__':
    model_name = 'runs/detect/train/weights/best.pt'
    model_data = 'SKU-110K.yaml'
    val(model_name,model_data)
