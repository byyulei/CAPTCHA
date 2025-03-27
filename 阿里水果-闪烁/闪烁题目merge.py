# -*-coding:utf-8 -*-

"""
# File       : 闪烁题目merge.py
# Time       : 2024/7/24 10:17
# Author     : 阿J
# version    : 2024
# Description: 
"""
from base64 import b64decode
import io
from PIL import Image
import numpy as np
import cv2
from ddddocr import DdddOcr
import requests
import re
from recognition_object import ocr_click
import random
import time

ocr = DdddOcr(import_onnx_path='my_ocr_new.onnx',charsets_path='charsets.json')
# ocr = DdddOcr(import_onnx_path='my_ocr.onnx',charsets_path='charsets.json')
# ocr = DdddOcr()

def merge_imgs(img1: str, img2: str) -> bytes:
    img1 = b64decode(img1.replace("data:image/png;base64,", "").encode())
    img2 = b64decode(img2.replace("data:image/png;base64,", "").encode())
    # 将字节数据转换为图像对象
    image1 = Image.open(io.BytesIO(img1))
    image2 = Image.open(io.BytesIO(img2))
    # 将图像转换为numpy数组
    array1 = np.array(image1)
    array2 = np.array(image2)
    # 确保两张图像的尺寸相同
    if array1.shape != array2.shape:
        raise ValueError("The dimensions of the two images do not match")
    # 取每个像素点的最大值
    max_array = np.maximum(array1, array2)
    # 将结果数组转换回图像
    result_image = Image.fromarray(max_array)
    # 将图像保存为字节
    img_byte_array = io.BytesIO()
    result_image.save(img_byte_array, format='JPEG')
    img_byte_array = img_byte_array.getvalue()
    with open('merge.png','wb') as f:
        f.write(img_byte_array)
    return img_byte_array


def get_que(img1,merge_img_bytes):
    nparr = np.frombuffer(b64decode(img1.replace("data:image/png;base64,", "")), np.uint8)
    original = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    height, width = original.shape[:2]
    x1 = 144
    x2 = 0
    for x in range(width - 1, 144, -1):
        if original[0, x] != original[0, x - 1]:
            x2 = x
            break
    print('截取题目坐标',x1, x2)

    nparr = np.frombuffer(merge_img_bytes, np.uint8)
    original = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    _, original = cv2.threshold(original, 202, 255, cv2.THRESH_BINARY)

    original = original[0:height, x1-1 :x2+1]

    cv2.imwrite('merge_deal.png', original)

    success, encoded_image = cv2.imencode('.jpg', original)
    # 输出图片二进制数据
    que = ocr.classification(encoded_image.tobytes()).replace('瑰', '魂').replace('蓝', '篮')
    # with open(f'E:\dddd_trainer-main\imgs\\{que}_{i}{int(time.time() * 10000)}.png', 'wb') as f:
    #     # with open(f'./que_imgs/{out_info}_{i}{int(time.time()*10000)}.png', 'wb') as f:
    #     f.write(encoded_image.tobytes())
    print(que)
    return que.split('个')


if __name__ == '__main__':
    headers = {
        "accept": "*/*",
        "accept-language": "zh-CN,zh;q=0.9",
        "cache-control": "no-cache",
        "pragma": "no-cache",
        "priority": "u=1, i",
        "referer": "https://scportal.taobao.com/quali_show.htm?spm=a1z10.1-c-s.0.0.34175249ZXLZDr&uid=2206833789551&qualitype=1",
        "sec-ch-ua": "\"Not/A)Brand\";v=\"8\", \"Chromium\";v=\"126\", \"Google Chrome\";v=\"126\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }
    session = requests.session()
    session.headers = headers

    num = 150
    flag = 0
    for i in range(num):
        res = session.get('https://scportal.taobao.com/quali_show.htm?spm=a1z10.1-c-s.0.0.34175249ZXLZDr&uid=2206833789551&qualitype=1').text

        NCAPPKEY = re.findall('"NCAPPKEY": "(.*?)",', res)[0]
        SECDATA = re.findall('"SECDATA": "(.*?)",', res)[0]
        NCTOKENSTR = re.findall('"NCTOKENSTR": "(.*?)",', res)[0]

        url = "https://scportal.taobao.com/quali_show.htm/_____tmd_____/newslidecaptcha"
        params = {
            "token": NCTOKENSTR,
            "appKey": NCAPPKEY,
            "x5secdata": SECDATA,
            "language": "cn",
            "v": "00736788154383{}".format(random.randint(1000, 10000))
        }
        res = session.get(url, params=params).json()['data']

        src = res['ques'].strip('MERGE|')
        img1, img2 = src.split('|')
        merge_img_bytes = merge_imgs(img1, img2)
        que = get_que(img1, merge_img_bytes)

        back_img = b64decode(res['imageData'].split('base64,')[-1])
        with open('img.png', 'wb') as f:
            f.write(back_img)

        ocr_inf = ocr_click(back_img)
        try:
            x,y = ocr_inf.get(que[1])
        except:
            print('----------------------识别异常---------------------------')
            # input()
            continue
        flag+=1
        print(que)
        print(ocr_inf)

        # # 标出识别位置
        # img = cv2.imread('img.png')
        # cv2.line(img, (x, 0), (x, img.shape[0]), (0, 0, 255), 2)
        #
        # cv2.imwrite('result.png', img)
        #
        # input()

    print(f'测试数：{num}，准确度:{round(flag/num,2)}')
