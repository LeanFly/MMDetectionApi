# -*- coding: utf-8 -*-
"""
@File    :   api.py
@Time    :   2023/07/17 15:40:09
@Author  :   leanfly 
@Version :   1.0
"""


import cv2
import mmcv
import numpy as np
from copy import deepcopy
from mmdet.apis import init_detector, inference_detector

from fastapi import FastAPI, responses
import uvicorn
import requests
import base64
from pydantic import BaseModel
import json

from io import BytesIO
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

config_file = "./configs/scnet/scnet_r50_fpn_1x_coco.py"  # 配置文件路径
checkpoint_file = "./checkpoints/scnet/scnet_r50_fpn_1x_coco-c3f09857.pth"  # 预训练模型加载路径
device = "cpu"  # cpu|gpu

# 构建模型
model = init_detector(config_file, checkpoint_file, device=device)

# coco_80_dict 字典，coco 数据集用于检测、分割的部分共有 80 个类别。
coco_80_dict = {
    "person": {"index": 0, "id": 1, "describe": "人"},
    "bicycle": {"index": 1, "id": 2, "describe": "自行车"},
    "car": {"index": 2, "id": 3, "describe": "车"},
    "motorcycle": {"index": 3, "id": 4, "describe": "摩托车"},
    "airplane": {"index": 4, "id": 5, "describe": "飞机"},
    "bus": {"index": 5, "id": 6, "describe": "公交车"},
    "train": {"index": 6, "id": 7, "describe": "火车"},
    "truck": {"index": 7, "id": 8, "describe": "卡车"},
    "boat": {"index": 8, "id": 9, "describe": "船"},
    "traffic light": {"index": 9, "id": 10, "describe": "红绿灯"},
    "fire hydrant": {"index": 10, "id": 11, "describe": "消防栓"},
    "stop sign": {"index": 11, "id": 13, "describe": "停车标志"},
    "parking meter": {"index": 12, "id": 14, "describe": "停车收费表"},
    "bench": {"index": 13, "id": 15, "describe": "板凳"},
    "bird": {"index": 14, "id": 16, "describe": "鸟"},
    "cat": {"index": 15, "id": 17, "describe": "猫"},
    "dog": {"index": 16, "id": 18, "describe": "狗"},
    "horse": {"index": 17, "id": 19, "describe": "马"},
    "sheep": {"index": 18, "id": 20, "describe": "羊"},
    "cow": {"index": 19, "id": 21, "describe": "牛"},
    "elephant": {"index": 20, "id": 22, "describe": "大象"},
    "bear": {"index": 21, "id": 23, "describe": "熊"},
    "zebra": {"index": 22, "id": 24, "describe": "斑马"},
    "giraffe": {"index": 23, "id": 25, "describe": "长颈鹿"},
    "backpack": {"index": 24, "id": 27, "describe": "背包"},
    "umbrella": {"index": 25, "id": 28, "describe": "雨伞"},
    "handbag": {"index": 26, "id": 31, "describe": "手提包"},
    "tie": {"index": 27, "id": 32, "describe": "领带"},
    "suitcase": {"index": 28, "id": 33, "describe": "手提箱"},
    "frisbee": {"index": 29, "id": 34, "describe": "飞盘"},
    "skis": {"index": 30, "id": 35, "describe": "雪橇"},
    "snowboard": {"index": 31, "id": 36, "describe": "滑雪板"},
    "sports ball": {"index": 32, "id": 37, "describe": "运动球"},
    "kite": {"index": 33, "id": 38, "describe": "风筝"},
    "baseball bat": {"index": 34, "id": 39, "describe": "棒球棒"},
    "baseball glove": {"index": 35, "id": 40, "describe": "棒球手套"},
    "skateboard": {"index": 36, "id": 41, "describe": "滑板"},
    "surfboard": {"index": 37, "id": 42, "describe": "冲浪板"},
    "tennis racket": {"index": 38, "id": 43, "describe": "网球拍"},
    "bottle": {"index": 39, "id": 44, "describe": "瓶"},
    "wine glass": {"index": 40, "id": 46, "describe": "酒杯"},
    "cup": {"index": 41, "id": 47, "describe": "杯"},
    "fork": {"index": 42, "id": 48, "describe": "叉子"},
    "knife": {"index": 43, "id": 49, "describe": "刀"},
    "spoon": {"index": 44, "id": 50, "describe": "汤匙"},
    "bowl": {"index": 45, "id": 51, "describe": "碗"},
    "banana": {"index": 46, "id": 52, "describe": "香蕉"},
    "apple": {"index": 47, "id": 53, "describe": "苹果"},
    "sandwich": {"index": 48, "id": 54, "describe": "三明治"},
    "orange": {"index": 49, "id": 55, "describe": "橙"},
    "broccoli": {"index": 50, "id": 56, "describe": "西兰花"},
    "carrot": {"index": 51, "id": 57, "describe": "胡萝卜"},
    "hot dog": {"index": 52, "id": 58, "describe": "热狗"},
    "pizza": {"index": 53, "id": 59, "describe": "披萨"},
    "donut": {"index": 54, "id": 60, "describe": "甜甜圈"},
    "cake": {"index": 55, "id": 61, "describe": "蛋糕"},
    "chair": {"index": 56, "id": 62, "describe": "椅子"},
    "couch": {"index": 57, "id": 63, "describe": "沙发"},
    "potted plant": {"index": 58, "id": 64, "describe": "盆栽植物"},
    "bed": {"index": 59, "id": 65, "describe": "床"},
    "dining table": {"index": 60, "id": 67, "describe": "餐桌"},
    "toilet": {"index": 61, "id": 70, "describe": "厕所"},
    "tv": {"index": 62, "id": 72, "describe": "电视"},
    "laptop": {"index": 63, "id": 73, "describe": "笔记本电脑"},
    "mouse": {"index": 64, "id": 74, "describe": "鼠标"},
    "remote": {"index": 65, "id": 75, "describe": "遥控器"},
    "keyboard": {"index": 66, "id": 76, "describe": "键盘"},
    "cell phone": {"index": 67, "id": 77, "describe": "手机"},
    "microwave": {"index": 68, "id": 78, "describe": "微波炉"},
    "oven": {"index": 69, "id": 79, "describe": "烤箱"},
    "toaster": {"index": 70, "id": 80, "describe": "烤面包机"},
    "sink": {"index": 71, "id": 81, "describe": "水槽"},
    "refrigerator": {"index": 72, "id": 82, "describe": "冰箱"},
    "book": {"index": 73, "id": 84, "describe": "书"},
    "clock": {"index": 74, "id": 85, "describe": "时钟"},
    "vase": {"index": 75, "id": 86, "describe": "花瓶"},
    "scissors": {"index": 76, "id": 87, "describe": "剪刀"},
    "teddy bear": {"index": 77, "id": 88, "describe": "泰迪熊"},
    "hair drier": {"index": 78, "id": 89, "describe": "吹风机"},
    "toothbrush": {"index": 79, "id": 90, "describe": "牙刷"},
}


# 待推理图像的原始 numpy.ndarray 对象
async def gen_result(imgUrl):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.43"
    }

    img_data = requests.get(imgUrl, headers=headers).content

    img = cv2.imdecode(np.asarray(bytearray(img_data), dtype=np.uint8), -1)
    img = cv2.imencode(".jpg", img)[1]
    img = np.array(img)
    img = cv2.imdecode(img, -1)

    img_raw = mmcv.imread(img)

    # 推理的结果
    result = inference_detector(model, img_raw)
    # print(result)
    return result


# 结果图像的 numpy.ndarray 对象
# img_result = model.show_result(img_raw, result)
# 将结果图像另存为文件
# model.show_result(img_raw, result, out_file="demo/demo_result.jpg")

# 拆解推理结果 result 数据，以方便我们提取抠图的结构存储到 coco_80_dict 字典。检测框 result[0] 的长度固定为 len(result[0]) == 80，因为 coco 数据集用于检测、分割的部分共有 80 个类别


def gen_img_detection(
    img_raw,
    img_detection_data_x1,
    img_detection_data_x2,
    img_detection_data_y1,
    img_detection_data_y2,
):
    # 检测目标，绘制边框
    img_detection = deepcopy(img_raw)
    for x_i in range(img_detection_data_x1, img_detection_data_x2):
        img_detection[img_detection_data_y1, x_i] = [0, 0, 255]  # 重置为 [0,0,255] 即红色像素点
        img_detection[img_detection_data_y2, x_i] = [0, 0, 255]

    for y_i in range(img_detection_data_y1, img_detection_data_y2):
        img_detection[y_i, img_detection_data_x1] = [0, 0, 255]
        img_detection[y_i, img_detection_data_x2] = [0, 0, 255]

    cv2.imwrite(f"demo/bottle_result_detection.jpg", img_detection)


def gen_img_segmentation(img_raw):
    # 读取第一个分割对象，将其可视化
    img_segmentation = deepcopy(img_raw)  # 深拷贝原始 numpy.ndarray 对象
    img_segmentation_data = coco_80_dict["bottle"]["segmentation"][0]  # 第1个板凳的分割对象
    img_h, img_w, img_chanel = img_raw.shape  # (height/高度, width/宽度, chanel/通道)
    for h_i in range(img_h):
        for w_i in range(img_w):
            # 通过分割对象的 bool 值来判断像素点是否需要去除(重置为 [255,255,255] 即纯白背景)
            if not img_segmentation_data[h_i, w_i]:
                img_segmentation[h_i, w_i] = [255, 255, 255]

    cv2.imwrite(f"demo/bottle_result_segmentation.jpg", img_segmentation)


def gen_target_png(img_raw):
    # 把现在的 3 通道图像数据改成 4 通道（添加透明度的通道），输出透明背景的图像文件
    img_segmentation_png_raw = deepcopy(img_raw)  # 深拷贝原始numpy.ndarray对象
    img_h, img_w, img_chanel = img_raw.shape  # (height/高度, width/宽度, chanel/通道)
    img_segmentation_png = np.full(
        (img_h, img_w, 4), 0, dtype="uint8"
    )  # 创建(图像高,图像宽,4通道)大小的numpy.ndarray对象

    img_segmentation_png_data = coco_80_dict["bottle"]["segmentation"][0]
    for h_i in range(img_h):
        for w_i in range(img_w):
            if img_segmentation_png_data[h_i, w_i]:
                img_segmentation_png[h_i, w_i] = np.append(
                    img_segmentation_png_raw[h_i, w_i], 255
                )


async def check_target(url: str, key: str):
    res = {"target": key, "score": "", "msg": ""}
    try:
        result = await gen_result(url)
    except:
        res["msg"] = res["msg"] + "获取目标异常；"
    # 拆解推理结果 result 数据，以方便我们提取抠图的结构存储到 coco_80_dict 字典

    for cc_key, cc_value in coco_80_dict.items():
        coco_80_dict[cc_key]["number"] = len(result[0][cc_value["index"]])
        coco_80_dict[cc_key]["detection"] = result[0][cc_value["index"]]
        coco_80_dict[cc_key]["segmentation"] = result[1][cc_value["index"]]

    # print("**********************")
    # img_detection = deepcopy(img_raw)  # 深拷贝原始 numpy.ndarray 对象
    try:
        img_detection_data = coco_80_dict[key]["detection"][0]  # 第1个目标的检测框
        print(f"目标置信度 = {img_detection_data[4]*100}%")  # 绘制比较麻烦，就直接打印
        score = f"{img_detection_data[4]*100:.2f}%"
        res["score"] = score
    except:
        res["msg"] = res["msg"] + f"没有检测到目标: {key}；"

    return res


async def gen_target_crop(imgUrl):
    res = {"b64": "", "score": "", "size": {}}
    ################ 图片 处理 ################
    # 待推理图像的原始 numpy.ndarray 对象

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.43"
    }

    img_data = requests.get(imgUrl, headers=headers).content

    img = cv2.imdecode(np.asarray(bytearray(img_data), dtype=np.uint8), -1)

    img = cv2.imencode(".jpg", img)[1]
    img = np.array(img)
    img = cv2.imdecode(img, -1)
    img_raw = mmcv.imread(img)
    # print(img_raw)
    img_h = img_raw.shape[0]  # 图像高度像素值
    img_w = img_raw.shape[1]  # 图像宽度像素值
    print(f"\n {img_h} {img_w} \n")

    # 推理的结果
    result = inference_detector(model, img_raw)
    # print(result)

    # 拆解推理结果 result 数据，以方便我们提取抠图的结构存储到 coco_80_dict 字典

    for cc_key, cc_value in coco_80_dict.items():
        coco_80_dict[cc_key]["number"] = len(result[0][cc_value["index"]])
        coco_80_dict[cc_key]["detection"] = result[0][cc_value["index"]]
        coco_80_dict[cc_key]["segmentation"] = result[1][cc_value["index"]]

    # print("**********************")
    # img_detection = deepcopy(img_raw)  # 深拷贝原始 numpy.ndarray 对象
    img_detection_data = coco_80_dict["bottle"]["detection"][0]  # 第1个目标的检测框

    # print("**********************")
    img_detection_data_x1 = int(img_detection_data[0])
    img_detection_data_y1 = int(img_detection_data[1])
    img_detection_data_x2 = int(img_detection_data[2])
    img_detection_data_y2 = int(img_detection_data[3])
    print(f"目标置信度 = {img_detection_data[4]*100}%")  # 绘制比较麻烦，就直接打印
    score = f"{img_detection_data[4]*100:.2f}%"
    # 把现在的 3 通道图像数据改成 4 通道（添加透明度的通道），输出透明背景的图像文件
    img_segmentation_png_raw = deepcopy(img_raw)  # 深拷贝原始numpy.ndarray对象
    img_h, img_w, img_chanel = img_raw.shape  # (height/高度, width/宽度, chanel/通道)
    img_segmentation_png = np.full(
        (img_h, img_w, 4), 0, dtype="uint8"
    )  # 创建(图像高,图像宽,4通道)大小的numpy.ndarray对象

    img_segmentation_png_data = coco_80_dict["bottle"]["segmentation"][0]
    for h_i in range(img_h):
        for w_i in range(img_w):
            if img_segmentation_png_data[h_i, w_i]:
                img_segmentation_png[h_i, w_i] = np.append(
                    img_segmentation_png_raw[h_i, w_i], 255
                )
    ### 裁剪图片 img[y:y+h, x:x+w]；OpenCV的坐标系原点在左上角 ###
    h = img_detection_data_y2 - img_detection_data_y1
    w = img_detection_data_x2 - img_detection_data_x1
    print(f"............{h} {w}..............")
    try:
        png_crop = img_segmentation_png[
            img_detection_data_y1 : (img_detection_data_y1 + h),
            img_detection_data_x1 : (img_detection_data_x1 + w),
        ]
    except:
        print("...............im 裁图出错................")
    try:
        im = cv2.imencode(".png", png_crop)[1]
    except:
        print("...............im 转换出错................")
    try:
        img_b64 = str(base64.b64encode(im))[2:-1]
    except:
        print("...............base64 转换出错................")
    res["b64"] = img_b64
    res["score"] = score
    res["size"]["width"] = w
    res["size"]["height"] = h
    return res


async def b64_to_cvimg(base64_code):
    # base64解码
    img_data = base64.b64decode(base64_code)
    # 转换为np数组
    img_array = np.fromstring(img_data, np.uint8)
    # 转换成opencv可用格式
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)

    return img


################# 开始构建 api #################
class Item(BaseModel):
    img1: str
    img2: str


class Target(BaseModel):
    url: str
    key: str


app = FastAPI()


@app.get("/genTarget")
async def main(url: str):
    try:
        res = await gen_target_crop(url)
        return res
    except:
        print(f"********** gen_target_crop 出错 **********")
        return "gen_target_crop 出错"


@app.post("/imgequal")
async def target_equal(item: Item):
    res = {"img1": "", "img2": "", "score": "", "msg": "", "imgb64": ""}
    url1 = item.img1
    url2 = item.img2

    res1 = await gen_target_crop(url1)
    res2 = await gen_target_crop(url2)
    res["img1"] = res1
    res["img2"] = res2

    #
    im1 = await b64_to_cvimg(res1["b64"])
    im2 = await b64_to_cvimg(res2["b64"])
    #
    w1 = res1["size"]["width"]
    h1 = res1["size"]["height"]
    w2 = res2["size"]["width"]
    h2 = res2["size"]["height"]

    if h1 > h2:
        im2 = cv2.resize(
            im2, dsize=(w1, h1), fx=1, fy=1, interpolation=cv2.INTER_LINEAR
        )
    if h1 < h2:
        im1 = cv2.resize(
            im1, dsize=(w2, h2), fx=1, fy=1, interpolation=cv2.INTER_LINEAR
        )

    ### 计算图片的差异化程度 ###
    # 像素点差异对比
    try:
        pixel_diff = cv2.absdiff(im1, im2)
    except:
        res["msg"] = res["msg"] + "像素点差异计算异常"
        pixel_diff = None
    # 结构相似度
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    ssim_score = ssim(gray1, gray2)

    ######### plt 可视化 #########
    # 图1
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY))
    plt.title("Image 1")
    # 图2
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY))
    plt.title("Image 2")
    # 差异化
    plt.subplot(1, 3, 3)
    if pixel_diff is not None:
        plt.imshow(pixel_diff, cmap="gray")
    plt.title(f"Pixel Difference\nStructural Similarity: {ssim_score:.2f}")

    ### 将 plt 可视化结果转成图片 base64 进行输出

    buffer = BytesIO()
    plt.savefig(buffer, format="PNG")
    equal_imb64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    res["imgb64"] = equal_imb64
    res["score"] = f"{ssim_score:.2f}"

    return responses.JSONResponse(content=(res))


@app.post("/detect")
async def detect(item: Target):
    url = item.url
    key = item.key
    res = await check_target(url=url, key=key)
    return res


@app.get("/getbottle")
async def get_bottle(url):
    res = await gen_target_crop(url)

    return res


if __name__ == "__main__":
    uvicorn.run(app="gen_target:app", host="0.0.0.0", port=17496, reload=True)
