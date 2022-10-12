import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

global count_right, count_left
count_left = 0
count_right = 0


def detect(save_img=False):
    global count_right, count_left

    # set some threds and bool variables
    open_threds = 0.9
    grey_threds = 180
    is_open = False

    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    if opt.rknn_mode == True:
        print('model convert to rknn_mode')
        from models.common_rk_plug_in import surrogate_silu, surrogate_hardswish
        from models import common
        for k, m in model.named_modules():
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
            if isinstance(m, common.Conv):  # assign export-friendly activations
                if isinstance(m.act, torch.nn.Hardswish):
                    m.act = torch.nn.Hardswish()
                elif isinstance(m.act, torch.nn.SiLU):
                    # m.act = torch.nn.SiLU()
                    m.act = surrogate_silu()
            # elif isinstance(m, models.yolo.Detect):
            #     m.forward = m.forward_export  # assign forward (optional)

            if isinstance(m, common.SPP):  # assign export-friendly activations
                ### best
                # tmp = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1) for i in range(2)])
                # m.m[0] = tmp
                # m.m[1] = tmp
                # m.m[2] = tmp
                ### friendly to origin config
                tmp = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1) for i in range(2)])
                m.m[0] = tmp
                tmp = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1) for i in range(4)])
                m.m[1] = tmp
                tmp = nn.Sequential(*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1) for i in range(6)])
                m.m[2] = tmp

        ### use deconv2d to surrogate upsample layer.
        # replace_one = torch.nn.ConvTranspose2d(model.model[10].conv.weight.shape[0],
        #                                        model.model[10].conv.weight.shape[0],
        #                                        (2, 2),
        #                                        groups=model.model[10].conv.weight.shape[0],
        #                                        bias=False,
        #                                        stride=(2, 2))
        # replace_one.weight.data.fill_(1)
        # replace_one.eval().to(device)
        # temp_i = model.model[11].i
        # temp_f = model.model[11].f
        # model.model[11] = replace_one
        # model.model[11].i = temp_i
        # model.model[11].f = temp_f

        # replace_one = torch.nn.ConvTranspose2d(model.model[14].conv.weight.shape[0],
        #                                        model.model[14].conv.weight.shape[0],
        #                                        (2, 2),
        #                                        groups=model.model[14].conv.weight.shape[0],
        #                                        bias=False,
        #                                        stride=(2, 2))
        # replace_one.weight.data.fill_(1)
        # replace_one.eval().to(device)
        # temp_i = model.model[11].i
        # temp_f = model.model[11].f
        # model.model[15] = replace_one
        # model.model[15].i = temp_i
        # model.model[15].f = temp_f

        ### use conv to surrogate slice operator
        from models.common_rk_plug_in import surrogate_focus
        surrogate_focous = surrogate_focus(int(model.model[0].conv.conv.weight.shape[1] / 4),
                                           model.model[0].conv.conv.weight.shape[0],
                                           k=tuple(model.model[0].conv.conv.weight.shape[2:4]),
                                           s=model.model[0].conv.conv.stride,
                                           p=model.model[0].conv.conv.padding,
                                           g=model.model[0].conv.conv.groups,
                                           act=True)
        surrogate_focous.conv.conv.weight = model.model[0].conv.conv.weight
        surrogate_focous.conv.conv.bias = model.model[0].conv.conv.bias
        surrogate_focous.conv.act = model.model[0].conv.act
        temp_i = model.model[0].i
        temp_f = model.model[0].f

        model.model[0] = surrogate_focous
        model.model[0].i = temp_i
        model.model[0].f = temp_f
        model.model[0].eval().to(device)

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    print('names', names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once #初始化模型
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        # size_ = (int(((img.shape[1] * 0.5) // 32) * 32), int(((img.shape[2]*0.5) // 32) * 32))
        # img_ = img.transpose(1, 2, 0)
        # img_ = cv2.resize(img_, size_)
        # # img_ = img_.transpose(2, 1, 0)
        # # img_b = np.zeros((416, 416, 3))
        # img_b = np.zeros_like(img)
        # img_b = img_b.transpose(2, 1, 0)
        #
        # img_b[0:size_[1], 0:size_[0]] = img_
        #
        # cv2.imwrite('img.jpg', img_b)
        # img_b = img_b.transpose(2, 0, 1)
        # h_scale = img.shape[0] / 192
        # w_scale = img.shape[1] / 192
        # img = torch.from_numpy(img_b).to(device)
        # img = img.transpose(1, 2, 0)

        # img_new = np.zeros((416, 416, 3))
        # short_len = min(img.shape[1], img.shape[2])
        #
        img = img.reshape((img.shape[1:]))
        # # h_ = img.shape[1]
        # # w_ = img.shape[2]
        # # short_idx = [1 if h_ > w_ else 0]
        #
        # # 256 416
        # size_ = (int(((img.shape[2] * 0.5) // 32) * 32), int(((img.shape[1] * 0.5) // 32) * 32))
        # offset = (img.shape[2] * 0.5 - size_[0], img.shape[1] * 0.5 - size_[1])
        #
        # img = img.transpose(1, 2, 0)
        #
        # img = cv2.resize(img, size_)
        #
        # h_ = img.shape[0]
        # w_ = img.shape[1]
        # short_idx = [1 if h_ > w_ else 0]
        #
        # img_new[0:h_, 0:w_] = img
        #
        # # img_new = cv2.resize(img, (416, 416))
        img = img.transpose(1, 2, 0)
        img = cv2.resize(img, (104, 104))
        img_new = np.zeros((416, 416, 3))
        img_new[0:104, 0:104] = img
        img = img_new.transpose(2, 0, 1)

        img = torch.from_numpy(img).to(device)

        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # 只检测框最大的

            if len(det):
                # Rescale boxes from img_size to im0 size
                # img_ = img_.transpose(2, 0, 1)
                # img_b = img_b.transpose(1, 2, 0)
                # img = img.permute(0, 2, 3, 1).contiguous()

                # det[:, 1] += int(offset[1] / 2)
                # det[:, 3] += int(offset[1] / 2)
                # det[:, 0] += int(offset[0] / 2)
                # det[:, 2] += int(offset[0] / 2)
                #
                # det[:, :4] *= 2
                #
                # ad = (416 - short_len) / 2
                # if short_idx[0] == 0:
                #     det[:, 1] += ad
                #     det[:, 3] += ad
                # else:
                #     det[:, 0] += ad
                #     det[:, 2] += ad

                # recover resize
                det[:, :4] *= 4
                scale_h = im0.shape[0] / 416
                scale_w = im0.shape[1] / 416
                det[:, 1] *= scale_h
                det[:, 3] *= scale_h
                det[:, 0] *= scale_w
                det[:, 2] *= scale_w

                # judge if mouth open or not by ratios
                mouth_w = det[:, 2] - det[:, 0]
                mouth_h = det[:, 3] - det[:, 1]
                open_ratio = mouth_h / mouth_w
                if open_ratio > open_threds:
                    is_open = True
                    cv2.putText(im0, 'open!!', (30, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255), 2)

                    # draw two gland bbx according to four points
                    color = (0, 0, 255)
                    unit_len = mouth_w / 12
                    left_gland_center = (det[:, 0], (det[:, 3] + det[:, 1]) / 2)
                    right_gland_center = (det[:, 2], (det[:, 3] + det[:, 1]) / 2)

                    # cv2.circle(im0, (int(det[:, 0]), int(det[:, 1])), 5, color, -1)
                    # cv2.circle(im0, (int(det[:, 2]), int(det[:, 3])), 5, color, -1)
                    # cv2.circle(im0, (int(det[:, 0]), int(det[:, 3])), 5, color, -1)
                    # cv2.circle(im0, (int(det[:, 2]), int(det[:, 1])), 5, color, -1)
                    #
                    # cv2.circle(im0, (int(left_gland_center[0]), int(left_gland_center[1])), 5, color, -1)
                    # cv2.circle(im0, (int(right_gland_center[0]), int(right_gland_center[1])), 5, color, -1)

                    left_gland_left_top = (
                        int(left_gland_center[0] + unit_len * 1.5), int(left_gland_center[1] - unit_len * 0.5))
                    left_gland_right_bott = (
                        int(left_gland_center[0] + unit_len * 2.5), int(left_gland_center[1] + unit_len * 0.5))
                    right_gland_left_top = (
                        int(right_gland_center[0] - unit_len * 2.5), int(right_gland_center[1] - unit_len * 0.5))
                    right_gland_right_bott = (
                        int(right_gland_center[0] - unit_len * 1.5), int(left_gland_center[1] + unit_len * 0.5))

                    # grey area detect
                    # 1: crop
                    gland_left_img = im0[left_gland_left_top[1]: left_gland_right_bott[1],
                                     left_gland_left_top[0]: left_gland_right_bott[0]]
                    gland_right_img = im0[right_gland_left_top[1]: right_gland_right_bott[1],
                                      right_gland_left_top[0]: right_gland_right_bott[0]]
                    # actual checking area crop
                    gland_right_img_checking = gland_right_img[0: int(0.5 * gland_right_img.shape[0]),
                                               int(0.25 * gland_right_img.shape[1]):int(
                                                   0.75 * gland_right_img.shape[1])]
                    gland_left_img_checking = gland_left_img[0: int(0.5 * gland_left_img.shape[0]),
                                              int(0.25 * gland_left_img.shape[1]):int(0.75 * gland_left_img.shape[1])]

                    # convert to grey after changing
                    grey_gland_right_img = cv2.cvtColor(gland_right_img_checking, cv2.COLOR_BGR2GRAY)
                    grey_gland_left_img = cv2.cvtColor(gland_left_img_checking, cv2.COLOR_BGR2GRAY)

                    # mean and max value before changing
                    mean_left = int(np.mean(grey_gland_left_img))
                    mean_right = int(np.mean(grey_gland_right_img))
                    max_left = int(np.max(grey_gland_left_img))
                    max_right = int(np.max(grey_gland_right_img))

                    # show the mean value in text type for checking
                    info_left_pixel = 'mean: ' + str(mean_left) + ', ' + 'max: ' + str(max_left)
                    info_right_pixel = 'mean: ' + str(mean_right) + ', ' + 'max: ' + str(max_right)

                    # write into file directory if mean value over a thred
                    path_right = './mouth_data/mouth/filtered_gland_right/'
                    if mean_right > grey_threds:
                        count_right += 1
                        path_right += str(mean_right) + '.jpg'
                        cv2.imwrite(path_right, gland_right_img)
                        # cv2.putText(inp_img, 'writing...', (300, 400), cv2.FONT_HERSHEY_SIMPLEX,
                        #             1, (0, 0, 0), 2)

                    path_left = './mouth_data/mouth/filtered_gland_left/'
                    if mean_left > grey_threds:
                        count_left += 1
                        path_left += str(mean_left) + '.jpg'
                        cv2.imwrite(path_left, gland_left_img)
                        # cv2.putText(inp_img, 'writing...', (100, 400), cv2.FONT_HERSHEY_SIMPLEX,
                        #             1, (0, 0, 0), 2)

                    # display some info
                    cv2.putText(im0, info_left_pixel, left_gland_left_top, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                1)
                    cv2.putText(im0, info_right_pixel, right_gland_left_top, cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 1)

                    # 如果检测到嘴部张开再画框--因为要进行灰度测试，框的rgb会影响灰度值，所以放到最后显示
                    cv2.rectangle(im0, left_gland_left_top, left_gland_right_bott, color, 2)
                    cv2.rectangle(im0, right_gland_left_top, right_gland_right_bott, color, 2)

                else:
                    is_open = False
                    cv2.putText(im0, 'close!!', (30, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255), 2)

                # print("open ratio is :       ")
                # print(open_ratio)

                # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # det[:, :4] = scale_coords(img_.shape[1:], det[:, :4], img.shape[1:]).round()
                # img = img.permute(0, 3, 1, 2).contiguous()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/mouth_best2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--rknn_mode', action='store_true', help='export rknn-friendly onnx model')
    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
