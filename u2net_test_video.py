import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
from PIL import ImageChops
import glob

from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

img_bg = io.imread("data/test_images/tokyo.jpg")
img_bg = Image.fromarray(img_bg)

async def save_output(image_name,pred,d_dir,width=None, height=None):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split("/")[-1]
    if not width and not height:
        image = io.imread(image_name)
        imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BICUBIC)
        # TODO: maybe make optional
        inv_mask = ImageChops.invert(imo)
        bg = ImageChops.multiply(inv_mask, img_bg)
        imo = ImageChops.multiply(Image.fromarray(image), imo)
        imo = ImageChops.add(imo, bg)
    else:
        imo = im.resize((width, height),resample=Image.BICUBIC)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.jpg')

def main():

    # --------- 1. get image path and name ---------
    model_name='u2netp'#u2netp


    image_dir = './data/workbench/'
    prediction_dir = './data/workbench_out/'
    model_dir = './saved_models/'+ model_name + '/' + model_name + '.pth'

    img_name_list = glob.glob(image_dir + '*')
    print(img_name_list)
    # TODO: consider data loader over sets of videos

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    batch_size = 1
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=3)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    from datetime import datetime
    a = datetime.now()
    total_inf = 0
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing frame:", i_test * batch_size)
        # print("dl:", datetime.now()-a)
        a = datetime.now()
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_test = inputs_test.cuda()

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        print("inf:", total_inf / (i_test + 1))
        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)
        total_inf += (datetime.now() - a).microseconds

        # save results to test_results folder
        # TODO: dynamically remember input sizes somehow, hardcoded for now
        for j in range(pred.shape[0]):
            save_output(img_name_list[batch_size * i_test + j],pred[j:j+1],prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7
        a = datetime.now()

if __name__ == "__main__":
    main()
