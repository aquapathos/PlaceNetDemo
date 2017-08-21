import chainer
import chainer.functions as F
from chainer import Variable
# from chainer.links import caffe
import PIL.Image
import numpy as np
import io
import urllib.request
import cv2
from chainer.links.model.vision.vgg import prepare as VGGprepare

from IPython.display import clear_output, Image, display
import pickle

mean = np.array([103.939, 116.779, 123.68])   # BGR
# blob データを PIL 画像に変換
def blob2img(blob, mean=mean):
    blob = (np.dstack(blob)+ mean)[:,:,::-1]   # BGR 2 RGB
    return PIL.Image.fromarray(np.uint8(blob))

# Google 翻訳サービスを使う　ーーーーーーーーーーーーーーーーーーー
import requests
import re
 
googleurl = 'https://translate.google.com/?hl=ja#en/ja/'
 
def translate(estring):
    r = requests.get(googleurl, params={'q': estring})
 
    pattern = "TRANSLATED_TEXT=\'(.*?)\'"
    jstring = re.search(pattern, r.text).group(1)
 
    return jstring
# Google 翻訳サービスを使う　ーーーーーーーーーーーーーーーーーーー

# 確率リストとしての出力からトップ５を出力するメソッドーーーーーーーーーーーーー
# ILSVSR１２のカテゴリデータ　　https://raw.githubusercontent.com/CSAILVision/places365/master/categories_hybrid1365.txt
categories = np.loadtxt("categories.txt",str,delimiter='\t')
def showtop(prob, ranklimit=5, trans=True): # prob は最終層から出力される確率リスト（Variable構造体)
    top5args = np.argsort(prob.data)[:-ranklimit-1:-1] # 上位５つの番号
    top5probs = prob.data[top5args] # 上位５つの確率
    for rank,(p,words) in enumerate(zip(top5probs,categories[top5args])):
        if trans:
            print(rank+1,translate(words[2:-1]))   #  on mac
        else:
            print(rank+1,words[2:-1])
        # print(rank,translate(words)
# 確率リストとしての出力からトップ５を出力するメソッドーーーーーーーーーーーーー

# caffemodelを読み込む
# model = caffe.CaffeFunction('modeldata/vgg16_hybrid1365.caffemodel')

vgg = pickle.load(open('modeldata/vgg16_hybrid1365.pkl', 'rb'))

def url2img(url):
    print(url)
    f = io.BytesIO(urllib.request.urlopen(url).read())
    img = PIL.Image.open(f)
    w,h = img.width, img.height
    if w > h:
        w1, h1 = int(224/h * w), 224
    else :
        w1,h1 = 224, int(224/w * h)
    return img.resize((w1,h1))
    
def predict(url="", trans=True):
    img = url2img(url)
    x = Variable( VGGprepare(img)[np.newaxis,])
    y, = vgg(inputs={'data': x}, outputs=['fc8a'])
    predict = F.softmax(y)
    showtop(predict[0], trans=trans)
    store(img)
    return img

nimg001 = np.zeros((224,224))
def nimg00():
    global nimg001
    return nimg001

def store(img):
     global nimg001
     nimg001 = np.asarray(img)
    