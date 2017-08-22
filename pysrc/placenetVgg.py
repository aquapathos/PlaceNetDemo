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
# blob データを PIL 画像に変
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
categories = np.loadtxt("pysrc/modeldata/categories.txt",str,delimiter='\t')
def showtop(prob, ranklimit=5, trans=True): # prob は最終層から出力される確率リスト（Variable構造体)
    top5args = np.argsort(prob.data)[:-ranklimit-1:-1] # 上位５つの番号
    top5probs = prob.data[top5args] # 上位５つの確率
    def numbercut(str):
        words = str[:-5] if str[-5]==' ' else str[:-4]
        refnum = str[-4:] if str[-5]==' ' else str[-3:]
        return words, refnum
    for rank,(p,words) in enumerate(zip(top5probs,categories[top5args])):
        word, refnum = numbercut(words[2:-1])
        if trans:
            print("{} {} ({})".format(rank+1,translate(word), refnum)) 
        else:
            print("{} {} ({})".format(rank+1,word, refnum))  
        # print(rank,translate(words)
# 確率リストとしての出力からトップ５を出力するメソッドーーーーーーーーーーーーー

# caffemodelを読み込む
# model = caffe.CaffeFunction('modeldata/vgg16_hybrid1365.caffemodel')

vgg = pickle.load(open('pysrc/modeldata/vgg16_hybrid1365.pkl', 'rb'))

def url2img(url):
    print(url)
    f = io.BytesIO(urllib.request.urlopen(url).read())
    img = PIL.Image.open(f)
    w,h = img.width, img.height
    if w > h:
        w1, h1 = int(448/h * w), 448
    else :
        w1,h1 = 448, int(448/w * h)
    return img.resize((w1,h1))
    
def predict(url="", trans=True):
    global pubimg
    if len(url) < 10 :  # おそらく操作ミスの場合
        return np.zeros((3,244,244))
    pubimg = url2img(url)
    x = Variable( VGGprepare(pubimg)[np.newaxis,])
    y, = vgg(inputs={'data': x}, outputs=['fc8a'])
    predict = F.softmax(y)
    showtop(predict[0], trans=trans)
    return pubimg