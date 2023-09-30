import flask
from flask import jsonify, request
import numpy as np
import time
from yolo5 import YOLOv5
import cv2


app = flask.Flask(__name__)
det = YOLOv5()


@app.route("/predict", methods=["POST"])
def predict():
    result = {"success": False}
    if request.method == "POST":
        if request.files.get("image") is not None:
            try:
                # 得到客户端传输的图像
                start = time.time()
                input_image = request.files["image"].read()
                imBytes = np.frombuffer(input_image, np.uint8)
                iImage = cv2.imdecode(imBytes, cv2.IMREAD_COLOR)
                print('1')
                # 执行推理
                outs = det.infer(iImage)
                print('2')
                print(outs)
                print("duration: ", time.time() - start)

                if (outs is None) and (len(outs) < 0):
                    result["success"] = False
                # 将结果保存为json格式
                result["box"] = outs[0].tolist()
                result["conf"] = outs[1].tolist()
                result["classid"] = outs[2].tolist()
                result['success'] = True

            except Exception:
                pass

    return jsonify(result)


if __name__ == "__main__":
    print(("* Loading model and Flask starting server..."
           "please wait until server has fully started"))
    app.run(host='127.0.0.1', port=7000)
