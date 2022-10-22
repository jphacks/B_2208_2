from firebase_admin import credentials, initialize_app, storage, firestore
import os
import uuid
import datetime

cred = credentials.Certificate(
    r"aipic-365801-firebase-adminsdk-i95fx-bc2e87bad9.json")
initialize_app(cred, {'storageBucket': 'gs://aipic-365801.appspot.com'})
db = firestore.client()


def sendPic(picPath):
    bucket = storage.bucket('aipic-365801.appspot.com')
    blob = bucket.blob(os.path.basename(picPath))
    blob.upload_from_filename(picPath)
    blob.make_public()
    print(blob.public_url)
    return blob.public_url


def insertGeneratedPicInfo(picInfo):
    try:
        doc_ref = db.collection(u'generated_pic').document(f'{uuid.uuid4()}')
        doc_ref.set({
            'prompts': picInfo['prompts'],
            'picUrl': picInfo['picUrl'],
            'creadedAt': picInfo['creadedAt'],
            'updatedAt': picInfo['updatedAt'],
        })
    except Exception as e:
        _outputError(e)


def insertRecommendedPicInfo():
    picInfos = []
    typeSubject = ["geometry Shapes", "building", "plant", "animal"]
    typeSpace = ["hotel like", "industrial", "west coast", "natural"]
    for num in range(4):
        picInfos.append({
            "category": "subject1",
            "picUrl": "",
            "type": typeSubject[num % 4],
        })
    for num in range(4):
        picInfos.append({
            "category": "subject2",
            "picUrl": "",
            "type": typeSubject[num % 4],
        })
    for num in range(4):
        picInfos.append({
            "category": "subject3",
            "picUrl": "",
            "type": typeSubject[num % 4],
        })
    for num in range(4):
        picInfos.append({
            "category": "space",
            "picUrl": "",
            "type": typeSpace[num % 4],
        })
    try:
        for picInfo in picInfos:
            dt_now = datetime.datetime.now()
            doc_ref = db.collection(
                u'recommended_pic').document(f'{uuid.uuid4()}')
            doc_ref.set({
                'category': picInfo['category'],
                'picUrl': picInfo['picUrl'],
                'type': picInfo['type'],
                'creadedAt': dt_now,
                'updatedAt': dt_now,
            })
    except Exception as e:
        _outputError(e)


def getRecommendedPicInfos():
    recommendedPicInfos = []
    try:
        docs = db.collection('recommended_pic').get()
        for doc in docs:
            recommendedPicInfos.append(doc.to_dict())
    except Exception as e:
        _outputError(e)
    return recommendedPicInfos


def _outputError(e):
    print('=== エラー内容 ===')
    print('type:' + str(type(e)))
    print('args:' + str(e.args))
    print('message:' + e.message)
    print('e自身:' + str(e))


if __name__ == '__main__':
    # insertRecommendedPicInfo()
    print(getRecommendedPicInfos())
