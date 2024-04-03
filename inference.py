from ultralytics import YOLO

def inference(video_name,show_labels = False,save = True):
    model = YOLO('runs/detect/train/weights/best.pt')
    results = model.track(source=video_name,save = save,stream=False,imgsz = 1080)
    if show_labels:
        labels = results[0].names
        for frame_num,result in enumerate(results):
            print(f'frame:{frame_num}')
            for obj_id,cls,conf in zip(result.boxes.id,result.boxes.cls,result.boxes.conf):
                obj_id,label,conf = int(obj_id.item()),labels[int(cls.item())],conf.item()
                print(f'ID:{obj_id},Class:{label},Confidence:{conf}')

if __name__ == '__main__':
    results = inference('Target Candy Shelf Organization w Crinkles 4-24-2023.mp4')
