# -*- coding: utf-8 -*-
import os
import random
import time
from io import BytesIO
from ultralytics.utils.plotting import Annotator
from orientation import non_max_suppression, tag_images
import numpy as np
from PIL import Image
import cv2
import onnxruntime
from 预测 import tb_match

np.set_printoptions(precision=4)

class YOLOV5_ONNX(object):
    def __init__(self,onnx_path, classes, providers=None):
        '''初始化onnx'''
        if not providers:
            providers = ['CPUExecutionProvider']
        self.onnx_session=onnxruntime.InferenceSession(onnx_path, providers=providers)
        # print(onnxruntime.get_device())
        self.input_name=self.get_input_name()
        self.output_name=self.get_output_name()
        self.classes=classes
        self.img_size = 320

    def get_input_name(self):
        '''获取输入节点名称'''
        input_name=[]
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)

        return input_name

    def get_output_name(self):
        '''获取输出节点名称'''
        output_name=[]
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)

        return output_name

    def get_input_feed(self,image_tensor):
        '''获取输入tensor'''
        input_feed={}
        for name in self.input_name:
            input_feed[name]=image_tensor

        return input_feed

    def letterbox(self,img, new_shape=(320, 320), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True,
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

    def clip_coords(self,boxes, img_shape):
        '''查看是否越界'''
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0].clip(0, img_shape[1])  # x1
        boxes[:, 1].clip(0, img_shape[0])  # y1
        boxes[:, 2].clip(0, img_shape[1])  # x2
        boxes[:, 3].clip(0, img_shape[0])  # y2

    def scale_coords(self,img1_shape, coords, img0_shape, ratio_pad=None):
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


    def infer(self,img_path):
        '''执行前向操作预测输出'''
        # 超参数设置
        img_size=(320,320) #图片缩放大小
        # 读取图片
        if isinstance(img_path, bytes):
            image = np.asarray(bytearray(BytesIO(img_path).read()), dtype=np.uint8)
            src_img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        else:
            src_img = cv2.imread(img_path)
        # start=time.time()

        src_size=src_img.shape[:2]

        # 图片填充并归一化
        img=self.letterbox(src_img,img_size,stride=32)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)


        # 归一化
        img=img.astype(dtype=np.float32)
        img/=255.0

        # # BGR to RGB
        # img = img[:, :, ::-1].transpose(2, 0, 1)
        # img = np.ascontiguousarray(img)

        # 维度扩张
        img=np.expand_dims(img,axis=0)
        # print('img resuming: ',time.time()-start)
        # 前向推理
        # start=time.time()
        input_feed=self.get_input_feed(img)
        # ort_inputs = {self.onnx_session.get_inputs()[0].name: input_feed[None].numpy()}
        pred = self.onnx_session.run(None, input_feed)[0]
        results = non_max_suppression(pred, 0.25,0.5)
        # print('onnx resuming: ',time.time()-start)
        # pred=self.onnx_session.run(output_names=self.output_name,input_feed=input_feed)
        # print(results)
        # "[tensor([[104.22188, 221.40257, 537.10876, 490.15454,   0.79675,   0.00000]])]"


        #映射到原始图像
        img_shape=img.shape[2:]
        # print(img_size)
        for det in results:  # detections per image
            if det is not None and len(det):
                det[:, :4] = self.scale_coords(img_shape, det[:, :4],src_size).round()
        # print(time.time()-start)
        # 输出评分
        if det is not None and len(det):
            self.draw(src_img, det)

    def plot_one_box(self,x, img, color=None, label=None, line_thickness=None):
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

    def draw(self,img, boxinfo):
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]
        for *xyxy, conf, cls in boxinfo:
            label = '%s %.2f' % (self.classes[int(cls)], conf)
            # print('目标检测坐标: ', xyxy)
            self.plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=1)

        # cv2.namedWindow("dst",0)
        # cv2.imshow("dst", img)
        cv2.imwrite("onnx_out.jpg",img)
        # cv2.waitKey(0)
        # cv2.imencode('.jpg', img)[1].tofile(os.path.join(dst, id + ".jpg"))
        return 0

    def new_draw(self,file, boxinfo):
        if isinstance(file,str):
            img = cv2.imread(file)  # BGR
        elif isinstance(file,bytes):
            img1 = np.frombuffer(file, np.uint8)
            img = cv2.imdecode(img1, cv2.IMREAD_ANYCOLOR)
        else:
            img = file

        for info in boxinfo:
            self.plot_one_box(info['crop'], img, label='', color=[random.randint(100, 255) for _ in range(3)], line_thickness=1)
        return img

    def decect(self, file):
        # 图片转换为矩阵
        if isinstance(file, np.ndarray):
            file_img = Image.fromarray(file)
        elif isinstance(file, bytes):
            file_img = Image.open(BytesIO(file))
        else:
            file_img = Image.open(file)
        file_img = file_img.convert('RGB')
        img = np.array(file_img)
        image_numpy = self.to_numpy(img, shape=(self.img_size, self.img_size))
        input_feed = self.get_input_feed(image_numpy)
        pred = self.onnx_session.run(None, input_feed)[0]
        pred = non_max_suppression(pred, 0.25, 0.5)
        # 输出评分
        res = tag_images(img, pred, self.img_size, self.classes, 0.7)

        # 画框
        # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]
        # for info in res:
        #     label = '%s %.2f' % (info['classes'], info['prob'])
        #     # print('目标检测坐标: ', xyxy)
        #     self.plot_one_box(info['crop'], img, label=label, color=colors[info['cls']], line_thickness=1)
        #
        # cv2.imwrite("out.jpg",img)

        # # 切割图片
        # for info in res:
        #     crop = info['crop']
        #     crop = file_img.crop(crop)
        #     crop.save(f'img/{int(time.time()*10000)}.jpg')

        return res

    def get_center(self, box):
        x1, y1, x2, y2 = box
        # 加点偏移值
        offset = 5
        return [int((x1 + x2) / 2)+offset, int((y1 + y2) / 2)+offset]



def ocr_click(img_path):
    # model.infer(img_path=img_path) # 目标检测
    s = model.decect(img_path)
    info = {}

    img = model.new_draw(img_path ,s)
    annotator = Annotator(img, example='识别', pil=True)
    # s左到右排序
    # print(s)
    s = sorted(s,key=lambda x:x['crop'][0])
    # print(s)
    for i,x in enumerate(s):
        # xy = model.get_center(x['crop'])
        xy = x['crop'][-2:]
        pic_img = img[x['crop'][1]:x['crop'][3], x['crop'][0]:x['crop'][2]]

        prob = tb_match(pic_img)
        name = list(prob.keys())[0]
        # print(name,x['crop'])

        # #判断识别的坐标是否在内
        # for q,o in enumerate(ocr_info):
        #     if x['crop'][0]<o[0] and x['crop'][2]>o[0] and x['crop'][1]<o[1] and x['crop'][3]>o[1]:
        #         # print(name,q,o,x['crop'])
        #         ocr_list[q] = name
        #         break
        # print(prob)
        # 查询某个文件夹下的文件数量
        file_path = r'E:\yolov5-master-jy-classify\datasets\data\train\\'+name
        file_nums = len(os.listdir(file_path))

        if file_nums < 100 :
            # print(name, file_nums)
            # 切割图片
            cv2.imwrite(f'{file_path}/{name}_{int(time.time() * 1000)}.png', pic_img)
        # if prob[name] < 0.2:
        #     print(f'过滤识别率文字{prob}')
        #     continue

        # if prob[name] < 0.6:
        #     print(prob)
        #     cv2.imwrite(f'{file_path}/{name}_{int(time.time() * 1000)}.jpg', pic_img)
            # 保存切割图
        # pic_img = cv2.resize(pic_img,(40,40))
        # cv2.imwrite(f'img/{int(time.time() * 1000)}.jpg', pic_img)


        info[name] = xy
        # 画出结果，随机颜色
        annotator.text([x['crop'][2], x['crop'][3]-20], name, txt_color=(0, 0, 255))

    # # txt = ''.join(ocr_list)
    # if len(txt) == 3 and txt not in xxxxx:
    #     print('大妈返回语序 》》 ',txt)
    #     with open('corpus.txt','a',encoding='utf-8') as f:
    #         f.write(txt+'\n')
    im0 = annotator.result()
    cv2.imwrite('tag_out.png', im0)
    return info


t1 = time.time()
model = YOLOV5_ONNX(onnx_path='阿里图标目标检测.onnx', classes=['box'])
print('加载目标检测模型耗时', time.time() - t1)



if __name__ == '__main__':

    # path = f'../data/images/'
    # for file in os.listdir(path):
    #     print(file)
    #     ocr_click(path+file)


    while True:
        res = ocr_click('img.png')
        print(res)
        input()

    # run(pic_img)
    # model.infer('tag_out.png')  # 目标检测
