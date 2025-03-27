import time
import torch
import torch.nn.functional as F
from models.common import DetectMultiBackend
from utils.augmentations import classify_transforms
from utils.dataloaders import  LoadImageContent
from utils.general import (Profile,cv2,)


def tb_match(source=None,imgsz=(128, 128),vid_stride=1,):
    # 加载模型
    stride, names, pt = model.stride, model.names, model.pt

    bs = 1  # batch_size

    # 识别
    dataset = LoadImageContent(source, img_size=imgsz, transforms=classify_transforms(imgsz[0]), vid_stride=vid_stride)

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup

    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    out_result = {}
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.Tensor(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            results = model(im)

        # Post-process
        with dt[2]:
            pred = F.softmax(results, dim=1)  # probabilities

        # Process predictions
        for i, prob in enumerate(pred):  # per image
            s += "%gx%g " % im.shape[2:]  # print string
            # annotator = Annotator(im0s.copy(), example=str(names), pil=True)

            # Print results
            top5i = prob.argsort(0, descending=True)[:1].tolist()  # top 5 indices
            s += f"{', '.join(f'{names[j]} {prob[j]:.2f}' for j in top5i)}, "

            # Write results
            # text = "\n".join(f"{prob[j]:.2f} {names[j]}" for j in top5i)

            # 赋值返回结果
            ooo = round(float(prob[top5i[0]]),2)
            # if ooo < 0.6:
            #     continue
            out_result[names[top5i[0]]] = ooo

            # # 画出结果
            # annotator.text([32, 32], text, txt_color=(255, 0, 0))
            #
            # # Stream results
            # im0 = annotator.result()
            #
            # # Save results (image with detections)
            # cv2.imwrite('./out.png', im0)

    return out_result


t1 = time.time()
model = DetectMultiBackend("阿里图标识别.pt")
print(f"模型加载耗时：{time.time() - t1:.2f}s")

if __name__ == "__main__":
    while True:
        result = tb_match(source="img.png")
        print(result)
        input()

    # with open("img.png",'rb') as f:
    #     result = run(source=f.read())
    #     print(result)

