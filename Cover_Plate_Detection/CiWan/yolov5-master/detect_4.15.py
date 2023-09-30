import argparse
import os
import sys
from pathlib import Path
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.utils import (cvtColor, letterbox_image,preprocess_input)
import openpyxl
import numpy as np
import torch
from PIL import Image
from nets import get_model_from_name


# 路径设置
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


# detect函数
@torch.no_grad()
def detect(
        weights=ROOT / 'runs/train/exp11/weights/last.pt',  # model.pt path(s)
        source=ROOT / 'dir',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/CW.yaml',  # dataset.yaml path
        conf_thres=0.01,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        imgsz=(640, 640),  # inference size (height, width)
        max_det=1000,  # maximum detections per image
        view_img=True,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        line_thickness=3,  # bou
        dnn=False,  # use OpenCV DNN for ONNX inferencending box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images

    save_dir = Path(project) / name
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '/n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"/n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

# 启用yolo模型进行预测
def detect_yolo():
    detect(
        weights=ROOT / 'weights/front134_2.pt',  # model.pt path(s)
        source=ROOT / 'C:/Users/ASUS/Desktop/Using AI/瓷碗目标检测/4.15小图+大图/20220414前盖面ok/Front/Crop/Front1',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/front134.yaml',  # dataset.yaml path
        conf_thres=0.2,  # confidence threshold
        iou_thres=0.1,  # NMS IOU threshold
        project=ROOT / 'runs/result_4.15/qiangai/front/crop',  # save results to project/name
        name='front1',  # save results to project/name
    )
    # detect(
    #     weights=ROOT / 'weights/front134_3.pt',  # model.pt path(s)
    #     source=ROOT / 'C:/Users/ASUS/Desktop/Using AI/瓷碗目标检测/4.15小图+大图/20220414前盖面ok/Front/Crop/Front3',  # file/dir/URL/glob, 0 for webcam
    #     data=ROOT / 'data/front134.yaml',  # dataset.yaml path
    #     conf_thres=0.25,  # confidence threshold
    #     iou_thres=0.1,  # NMS IOU threshold
    #     project=ROOT / 'runs/result_4.15/qiangai/front/crop',  # save results to project/name
    #     name='front3',  # save results to project/name
    # )
    # detect(
    #     weights=ROOT / 'weights/front134_3.pt',  # model.pt path(s)
    #     source=ROOT / 'C:/Users/ASUS/Desktop/Using AI/瓷碗目标检测/4.15小图+大图/20220414前盖面ok/Front/Crop/Front4',  # file/dir/URL/glob, 0 for webcam
    #     data=ROOT / 'data/front134.yaml',  # dataset.yaml path
    #     conf_thres=0.2,  # confidence threshold
    #     iou_thres=0.1,  # NMS IOU threshold
    #     project=ROOT / 'runs/result_4.15/qiangai/front/crop',  # save results to project/name
    #     name='front4',  # save results to project/name
    # )
    # detect(
    #     weights=ROOT / 'weights/front2_best.pt',  # model.pt path(s)
    #     source=ROOT / 'C:/Users/ASUS/Desktop/Using AI/瓷碗目标检测/4.15小图+大图/20220414前盖面ok/Front/Crop/Front2',  # file/dir/URL/glob, 0 for webcam
    #     data=ROOT / 'data/front2.yaml',  # dataset.yaml path
    #     conf_thres=0.2,  # confidence threshold
    #     iou_thres=0.1,  # NMS IOU threshold
    #     project=ROOT / 'runs/result_4.15/qiangai/front/crop',  # save results to project/name
    #     name='front2',  # save results to project/name
    # )

# 读取前盖label统计过杀率
def evaluate():
    guosha1 = len(os.listdir('runs/result_4.15/qiangai/front/crop/front1/labels')) / (len(os.listdir('runs/result_4.15/qiangai/front/crop/front1'))-1)
    # guosha2 = len(os.listdir('runs/result_4.15/qiangai/front/crop/front2/labels')) / (len(os.listdir('runs/result_4.15/qiangai/front/crop/front2'))-1)
    # guosha3 = len(os.listdir('runs/result_4.15/qiangai/front/crop/front3/labels')) / (len(os.listdir('runs/result_4.15/qiangai/front/crop/front3'))-1)
    # guosha4 = len(os.listdir('runs/result_4.15/qiangai/front/crop/front4/labels')) / (len(os.listdir('runs/result_4.15/qiangai/front/crop/front4'))-1)
    return (guosha1)
# , guosha2, guosha3, guosha4
if __name__ == '__main__':
    detect_yolo()
    print(evaluate())
