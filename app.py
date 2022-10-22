from flask import Flask, request
import generate_pic
import firebase_operator as fo
import datetime

import json

app = Flask(__name__)


@app.route('/')
def hello():
    return '画像自動生成API: http://3.113.209.61/api-gp?prompts=生成に使用する文字列 <br> アンケート画像取得用API: http://3.113.209.61/api-rp'


@app.route('/api-gp')
def generate():
    contents = request.args.get('prompts', '')
    # contents = "A painting, fingers-crossed, " + contents
    print(contents)
    prompts = [contents]
    picPath = generate_pic.generatePic(prompts)
    picUrl = fo.sendPic(picPath)
    dt_now = datetime.datetime.now()
    responseData = {
        'prompts': prompts,
        'picUrl': picUrl,
        'creadedAt': dt_now,
        'updatedAt': dt_now,
    }
    fo.insertGeneratedPicInfo(responseData)
    return json.dumps(responseData, default=str)


@app.route('/api-rp')
def getRecommendedPic():
    recommendedPicInfos = fo.getRecommendedPicInfos()
    return json.dumps(recommendedPicInfos, default=str)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
