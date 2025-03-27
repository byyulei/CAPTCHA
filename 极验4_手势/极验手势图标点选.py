# -*-coding:utf-8 -*-

"""
# File       : 图片获取.py
# Time       : 2024/4/12 13:59
# Author     : zhangwj
# version    : 2024
# Description: 
"""
import base64
import hashlib
import json
import re
import uuid

import execjs
import requests
from loguru import logger
from binascii import b2a_hex
from Crypto.Cipher import AES
from Crypto.Cipher import PKCS1_v1_5
from Crypto.PublicKey import RSA
from Crypto.Util.Padding import pad
import torch
import torch.nn.functional as F

# models,utils包在yolo5文件中，自行下载
from models.common import DetectMultiBackend
from utils.augmentations import classify_transforms
from utils.dataloaders import LoadImageContent
from utils.general import (Profile, )
import random
import time
from io import BytesIO
import numpy as np
from PIL import Image
import cv2
import onnxruntime

np.set_printoptions(precision=4)


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def tag_images(imgs, img_detections, img_size, classes, max_prob=0.6):  # max_prob 栓选预测的阈值
    imgs = [imgs]
    """图片展示"""
    results = []
    zero = lambda x: int(x) if x > 0 else 0
    for img_i, (img, detections) in enumerate(zip(imgs, img_detections)):
        # Create plot
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, img_size, img.shape[:2])
            for x1, y1, x2, y2, conf, cls_pred in detections:
                if conf > max_prob:
                    results.append(
                        {
                            "crop": [zero(i) for i in (x1, y1, x2, y2)],
                            "classes": classes[int(cls_pred)],
                            'prob': conf,
                            'cls': int(cls_pred),
                        }
                    )
        else:
            print("识别失败")
    return results


# 识别结果解析
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def nms(dets, scores, thresh):
    """Pure Python NMS baseline."""
    # x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]  # xmin
    y1 = dets[:, 1]  # ymin
    x2 = dets[:, 2]  # xmax
    y2 = dets[:, 3]  # ymax
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # argsort()返回数组值从小到大的索引值
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:  # 还有数据
        i = order[0]
        keep.append(i)
        if order.size == 1: break
        # 计算当前概率最大矩形框与其他矩形框的相交框的坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交框的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        IOU = inter / (areas[i] + areas[order[1:]] - inter)
        left_index = (np.where(IOU <= thresh))[0]
        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[left_index + 1]

    return np.array(keep)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

        Returns:
             list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres  # candidates
    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    t = time.time()
    output = [np.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = np.zeros((len(l), nc + 5))
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = np.concatenate((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero()
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None]), 1)
        else:  # best class only
            conf = x[:, 5:].max(1, keepdims=True)
            j = x[:, 5:].argmax(1)
            j = np.expand_dims(j, 0).T
            x = np.concatenate((box, conf, j), 1)[conf.reshape(1, -1)[0] > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        i = nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
        return output


class YOLOV5_ONNX(object):
    def __init__(self, onnx_path, classes, providers=None):
        '''初始化onnx'''
        if not providers:
            providers = ['CPUExecutionProvider']
        self.onnx_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        # print(onnxruntime.get_device())
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()
        self.classes = classes
        self.img_size = 320

    def get_input_name(self):
        '''获取输入节点名称'''
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)

        return input_name

    def get_output_name(self):
        '''获取输出节点名称'''
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)

        return output_name

    def get_input_feed(self, image_tensor):
        '''获取输入tensor'''
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = image_tensor

        return input_feed

    def letterbox(self, img, new_shape=(320, 320), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True,
                  stride=32):
        '''图片归一化'''
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def clip_coords(self, boxes, img_shape):
        '''查看是否越界'''
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0].clip(0, img_shape[1])  # x1
        boxes[:, 1].clip(0, img_shape[0])  # y1
        boxes[:, 2].clip(0, img_shape[1])  # x2
        boxes[:, 3].clip(0, img_shape[0])  # y2

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        '''
        坐标对应到原始图像上，反操作：减去pad，除以最小缩放比例
        :param img1_shape: 输入尺寸
        :param coords: 输入坐标
        :param img0_shape: 映射的尺寸
        :param ratio_pad:
        :return:
        '''

        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new,计算缩放比率
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
                    img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding ，计算扩充的尺寸
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding，减去x方向上的扩充
        coords[:, [1, 3]] -= pad[1]  # y padding，减去y方向上的扩充
        coords[:, :4] /= gain  # 将box坐标对应到原始图像上
        self.clip_coords(coords, img0_shape)  # 边界检查
        return coords

    def to_numpy(self, img, shape):
        # 超参数设置
        img_size = shape
        # img_size = (640, 640)  # 图片缩放大小
        # 读取图片
        # src_img = cv2.imread(img_path)
        src_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # start = time.time()
        # src_size = src_img.shape[:2]

        # 图片填充并归一化
        img = self.letterbox(src_img, img_size, stride=32)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # 归一化
        img = img.astype(dtype=np.float32)
        img /= 255.0

        # # BGR to RGB
        # img = img[:, :, ::-1].transpose(2, 0, 1)
        # img = np.ascontiguousarray(img)

        # 维度扩张
        img = np.expand_dims(img, axis=0)
        # print('img resuming: ', time.time() - start)
        # 前向推理
        # start=time.time()

        return img

    def infer(self, img_path):
        '''执行前向操作预测输出'''
        # 超参数设置
        img_size = (320, 320)  # 图片缩放大小
        # 读取图片
        if isinstance(img_path, bytes):
            image = np.asarray(bytearray(BytesIO(img_path).read()), dtype=np.uint8)
            src_img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if isinstance(img_path, str):
            src_img = cv2.imread(img_path)
        else:
            src_img = cv2.cvtColor(np.array(img_path), cv2.COLOR_RGB2BGR)
        # start=time.time()

        src_size = src_img.shape[:2]

        # 图片填充并归一化
        img = self.letterbox(src_img, img_size, stride=32)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # 归一化
        img = img.astype(dtype=np.float32)
        img /= 255.0

        # # BGR to RGB
        # img = img[:, :, ::-1].transpose(2, 0, 1)
        # img = np.ascontiguousarray(img)

        # 维度扩张
        img = np.expand_dims(img, axis=0)
        # print('img resuming: ',time.time()-start)
        # 前向推理
        # start=time.time()
        input_feed = self.get_input_feed(img)
        # ort_inputs = {self.onnx_session.get_inputs()[0].name: input_feed[None].numpy()}
        pred = self.onnx_session.run(None, input_feed)[0]
        results = non_max_suppression(pred, 0.5, 0.5)
        # print('onnx resuming: ',time.time()-start)
        # pred=self.onnx_session.run(output_names=self.output_name,input_feed=input_feed)
        # print(results)
        # "[tensor([[104.22188, 221.40257, 537.10876, 490.15454,   0.79675,   0.00000]])]"

        # 映射到原始图像
        img_shape = img.shape[2:]
        # print(img_size)
        for det in results:  # detections per image
            if det is not None and len(det):
                det[:, :4] = self.scale_coords(img_shape, det[:, :4], src_size).round()
        # print(time.time()-start)
        # 输出评分
        if det is not None:
            self.draw(src_img, det)

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def draw(self, img, boxinfo):
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]
        i = 1
        for *xyxy, conf, cls in boxinfo:
            label = '%s %.2f' % (self.classes[int(cls)], conf)
            self.plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=2)

        # cv2.namedWindow("dst",0)
        # cv2.imshow("dst", img)
        cv2.imwrite("draw.jpg", img)
        # cv2.waitKey(0)
        # cv2.imencode('.jpg', img)[1].tofile(os.path.join(dst, id + ".jpg"))
        return 0

    def decect(self, file):
        # 图片转换为矩阵
        if isinstance(file, np.ndarray):
            img = Image.fromarray(file)
        elif isinstance(file, bytes):
            img = Image.open(BytesIO(file))
        else:
            img = Image.open(file)
        img = img.convert('RGB')
        img = np.array(img)
        image_numpy = self.to_numpy(img, shape=(self.img_size, self.img_size))
        input_feed = self.get_input_feed(image_numpy)
        pred = self.onnx_session.run(None, input_feed)[0]
        pred = non_max_suppression(pred, 0.25, 0.45)
        # 输出评分
        res = tag_images(img, pred, self.img_size, self.classes, 0.6)

        # 画框
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]
        for info in res:
            label = '%s %.2f' % (info['classes'], info['prob'])
            # print('目标检测坐标: ', xyxy)
            self.plot_one_box(info['crop'], img, label=label, color=colors[info['cls']], line_thickness=1)

        # cv2.imwrite("out.jpg",img)

        return res

    def get_center(self, box):
        x1, y1, x2, y2 = box
        return [int((x1 + x2) / 2), int((y1 + y2) / 2)]


def open_image(file):
    # 图片转换为矩阵
    if isinstance(file, np.ndarray):
        img = Image.fromarray(file)
    elif isinstance(file, bytes):
        img = Image.open(BytesIO(file))
    else:
        img = Image.open(file)
    return img


def drow_img(img_path, result):
    if isinstance(img_path, bytes):
        image = np.asarray(bytearray(BytesIO(img_path).read()), dtype=np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(img_path)

    def plot_one_box(x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return img

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(result))]
    for i, xyxy in enumerate(result):
        label = i + 1
        img = plot_one_box(xyxy, img, label=str(label), color=colors[i], line_thickness=2)
    cv2.imwrite("out.jpg", img)


def tb_match(source=None, imgsz=(128, 128), vid_stride=1, ):
    # 加载模型
    stride, names, pt = fl_model.stride, fl_model.names, fl_model.pt

    bs = 1  # batch_size

    # 识别
    dataset = LoadImageContent(source, img_size=imgsz, transforms=classify_transforms(imgsz[0]), vid_stride=vid_stride)

    # Run inference
    fl_model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup

    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    out_result = {}
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.Tensor(im).to(fl_model.device)
            im = im.half() if fl_model.fp16 else im.float()  # uint8 to fp16/32
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            results = fl_model(im)

        # Post-process
        with dt[2]:
            pred = F.softmax(results, dim=1)  # probabilities

        for i, prob in enumerate(pred):  # per image
            s += "%gx%g " % im.shape[2:]  # print string
            top5i = prob.argsort(0, descending=True)[:1].tolist()  # top 5 indices
            s += f"{', '.join(f'{names[j]} {prob[j]:.2f}' for j in top5i)}, "
            # 赋值返回结果
            out_result[names[top5i[0]]] = round(float(prob[top5i[0]]), 2)

    return out_result


def S4():
    return "{:04x}".format(random.randint(0, 65535))


def get_key():
    return S4() + S4() + S4() + S4()


def md5(s):
    m = hashlib.md5()
    m.update(s.encode('utf-8'))
    return m.hexdigest()


def rsaEncryptByModule(content, module, pubKey="10001", encode="BASE64"):
    """

    :param content: 待加密文本
    :param module: HEX 编码的 module
    :param pubKey: 默认 10001
    :param encode: BASE64/HEX
    :return:
    """
    pubKey = int(pubKey, 16)
    modules = int(module, 16)
    pubobj = RSA.construct((modules, pubKey), False)
    public_key = pubobj.publickey().exportKey().decode()
    rsakey = RSA.importKey(public_key)
    rsa = PKCS1_v1_5.new(rsakey)
    if encode == "BASE64":
        cipher_text = base64.b64encode(rsa.encrypt(content.encode('utf-8')))
    else:
        cipher_text = b2a_hex(rsa.encrypt(content.encode('utf-8')))
    return cipher_text.decode('utf-8')


def bytes_to_str(arr):
    return ''.join(['{:02x}'.format(b) for b in arr])


def aes_encrypt_jy(data, key, iv='0000000000000000'):
    cipher = AES.new(key.encode("utf-8"), AES.MODE_CBC, iv.encode("utf-8"))
    padded_data = pad(data.encode("utf-8"), AES.block_size)
    ciphertext = cipher.encrypt(padded_data)
    return bytes_to_str(ciphertext)


def enc_w(click_list, captcha_id, lot_number, pow_detail,w_params={}):
    aeskey = get_key()
    pow_msg = pow_detail['version'] + "|" + str(pow_detail['bits']) + "|" + pow_detail['hashfunc'] + "|" + pow_detail[
        'datetime'] + "|" + captcha_id + "|" + lot_number + "||" + aeskey

    o = {
        "passtime": random.randint(1000, 1500),
        "userresponse": click_list,
        "device_id": "",
        "lot_number": lot_number,
        "pow_msg": pow_msg,
        "pow_sign": md5(pow_msg),
        "geetest": "captcha",
        "lang": "zh",
        "ep": "123",
        # "biht": "1426265548",
        # "vYTv": "Z4XP",
        "em": {"ph": 0, "cp": 0, "ek": "11", "wd": 1, "nt": 0, "si": 0, "sc": 0}
    }
    o.update(w_params)
    _ = rsaEncryptByModule(aeskey,
                           "00C1E3934D1614465B33053E7F48EE4EC87B14B95EF88947713D25EECBFF7E74C7977D02DC1D9451F79DD5D1C10C29ACB6A9B4D6FB7D0A0279B6719E1772565F09AF627715919221AEF91899CAE08C0D686D748B20A3603BE2318CA6BC2B59706592A9219D0BF05C9F65023A21D2330807252AE0066D59CEEFA5F2748EA80BAB81",
                           encode='HEX')
    l = aes_encrypt_jy(json.dumps(o, separators=(',', ':')), aeskey)
    return l + _



def get_gct(js_url):
    xxxxx = requests.get(url=js_url).text

    # 匹配函数名
    zzzzz = re.findall("function (.*?)\(t\){var ", xxxxx, re.S)[1]

    FFFF = str(re.findall("function (.*?)\(t\){var ", xxxxx, re.S)[0]).split(" ")[-1]
    new_zzzz = "function " + zzzzz + "(t)"
    new_FFFF = "function " + FFFF + "(t)"
    new_xxxxx = xxxxx.replace(new_zzzz, ";zzzz=" + new_zzzz).replace(new_FFFF, "FFFF=" + new_FFFF).replace(
        "return function(t)", ";return function(t)").replace("'use strict';var ", "dtcg=")

    new_js = r"""window=this;
    var kkkk;var dtcg;var FFFF;
    var zzzz;""" + new_xxxxx + r"""
    function getct(){
    return [dtcg,FFFF(zzzz.toString()+FFFF(FFFF.toString()))] }
    """
    # print(new_js)
    ppp = execjs.compile(new_js).call('getct')
    return {ppp[0]:ppp[1]}


def get_gcaptcha4_param(js_url):
    js_txt = requests.get(js_url).text
    new_js = js_txt.split('(){}!function()')[0]+'(){}'
    # print(new_js)

    # 匹配解密参数的函数名
    zzz = re.findall(";function (.*?)\(\)\{\}", js_txt)[0]
    ppppp = re.findall(zzz+"\.(.{4})=function", js_txt)[2]

    ctx = execjs.compile(new_js)

    xxx = re.findall(']=\{(.*?)};}\(\),function webpackUniversalModuleDefinition',js_txt)[0].split(':')
    key = eval(xxx[0])
    num = re.findall('\d+',xxx[1])[0]

    value = ctx.eval(zzz+'.'+ppppp+'('+num+')')
    return {key:value}

def get_param(c_url, g_url):
    p1 = get_gcaptcha4_param(c_url)
    p2 = get_gct(g_url)
    p1.update(p2)
    return p1

def ocr_tb():
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.82 Safari/537.36',
        'referer': 'https://gt4.geetest.com/',
    }

    data = {
        "captcha_id": 'd01910e7bfdb40a4af221686f30ad41c',
        "challenge": uuid.uuid4(),
        "client_type": "web",
        "lang": "zho"
    }
    res = requests.get('https://gcaptcha4.geetest.com/load', data=data, headers=headers).text
    res = json.loads(res[1:-1])
    g_url = 'https://static.geetest.com'+res['data']['gct_path']
    c_url = 'https://static.geetest.com'+res['data']['static_path']+res['data']['js']
    global w_params
    if not w_params:
        w_params = get_param(c_url, g_url)
        logger.warning('获取动态参数 {}'.format(w_params))

    img_buffer = requests.get('https://static.geetest.com/{}'.format(res['data']['imgs'])).content
    ques = [que.split('/')[-1].split('.')[0] for que in res['data']['ques']]
    logger.info('需要点击的图片 {}'.format(ques))

    lot_number = res['data']['lot_number']
    payload = res['data']['payload']
    process_token = res['data']['process_token']
    pow_detail = res['data']['pow_detail']

    arr = np.frombuffer(img_buffer, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    pos_info = model.decect(img_buffer)
    match_pos = []
    macth_dict = {}
    for pos in pos_info:
        pic_img = img[pos['crop'][1]:pos['crop'][3], pos['crop'][0]:pos['crop'][2]]
        x1, y1, x2, y2 = pos['crop']
        center = [int((x1 + x2) / 2 / 301 * 10000), int((y1 + y2) / 2 / 201 * 10000)]
        fl_match = tb_match(pic_img)
        key = list(fl_match.keys())[0]
        # if fl_match[key]<0.4: # 去除异常识别
        #     continue
        macth_dict[key] = center

    logger.info('识别到的图片 {}'.format(macth_dict))
    if len(macth_dict) < 3:
        logger.error('识别失败')
        return

    for que in ques:
        pos_info = macth_dict.get(que)
        match_pos.append(pos_info)
        if pos_info:
            macth_dict.pop(que)

    v_list = list(macth_dict.values())
    if len(v_list) <= 1:  # 模板匹配用的其他模型，可能会出现多一个匹配的情况，自适应提高成功率
        if len(v_list) == 1:
            logger.warning('自适应调整 {}'.format(macth_dict))
            for pos in match_pos:
                if not pos:
                    match_pos[match_pos.index(pos)] = v_list[0]
    else:
        raise Exception('匹配失败')

    logger.debug('提交坐标 {}'.format(match_pos))

    w = enc_w(match_pos, data['captcha_id'], lot_number, pow_detail,w_params)

    params = {
        'captcha_id': data['captcha_id'],
        'client_type': 'web',
        'lot_number': lot_number,
        'payload': payload,
        'process_token': process_token,
        'payload_protocol': '1',
        'pt': '1',
        'w': w,
        'callback': '',
    }
    res = requests.get('https://gcaptcha4.geetest.com/verify', params=params).text[1:-1]
    res = json.loads(res)['data']
    if res['result'] == 'success':
        logger.debug('验证成功 {}'.format(res))
    else:
        logger.error('验证失败 {}'.format(res))


if __name__ == '__main__':
    t1 = time.time()
    model = YOLOV5_ONNX(onnx_path='极验3图标目标检测.onnx', classes=['char', 'target'])
    logger.debug(f"加载目标检测模型耗时：{time.time() - t1:.2f}s")

    t1 = time.time()
    fl_model = DetectMultiBackend("best.pt")
    logger.debug(f"加载分类模型加载耗时：{time.time() - t1:.2f}s")

    w_params = {}

    while True:
        for i in range(10):
            ocr_tb()
        input()
