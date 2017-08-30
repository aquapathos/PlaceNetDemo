# import chainer
import chainer.functions as F
from chainer import Variable
import PIL.Image
import numpy as np
import io
import urllib.request

from chainer.links.model.vision.vgg import prepare as VGGprepare
import pickle

vgg = pickle.load(open('pysrc/modeldata/vgg16_hybrid1365.pkl', 'rb'))

mean = np.array([103.939, 116.779, 123.68])   # BGR
# blob データを PIL 画像に変
def blob2img(blob, mean=mean):
    blob = (np.dstack(blob)+ mean)[:,:,::-1]   # BGR 2 RGB
    return PIL.Image.fromarray(np.uint8(blob))

# 確率リストとしての出力からトップ５を出力するメソッド2
# 日本語化済みのカテゴリリストを用いる
f = open("pysrc/modeldata/jcategories.txt",'r',encoding="utf-8")
jcategories={}
for n in range(1365):
    jcategories[n]=f.readline()[:-1]
f.close()

def showtop2(prob, ranklimit=5): # prob は最終層から出力される確率リスト（Variable構造体)
    top5args = np.argsort(prob.data)[:-ranklimit-1:-1] # 上位５つの番号
    top5probs = prob.data[top5args] # 上位５つの確率
    for rank,(n, p) in enumerate(zip(top5args,top5probs)):
        print("{} {} ({:7.5f})".format(rank+1,jcategories[n], top5probs[rank]))

def url2img(url):
    # print(url)
    if url[:16] == "http://localhost":
        pic = url.rsplit('/',1)[1]
        f = open("pics/"+pic,'rb')
    elif url[:4] != "http":
        f = open(url,'rb')
    else:
        f = io.BytesIO(urllib.request.urlopen(url).read())
    img = PIL.Image.open(f)
    w,h = img.width, img.height
    if w > h:
        w1, h1 = int(448/h * w), 448
    else :
        w1,h1 = 448, int(448/w * h)
    return img.resize((w1,h1))

def predict(url=""):
    global pubimg
    if len(url) < 10 :  # おそらく操作ミスの場合
        return np.zeros((3,244,244))
    pubimg = url2img(url)
    x = Variable( VGGprepare(pubimg)[np.newaxis,])
    y, = vgg(inputs={'data': x}, outputs=['fc8a'])
    predict = F.softmax(y)
    showtop2(predict[0])
    return pubimg
