#Saral Bhagat
import cv2
import numpy as np

import sys

def convert_color_space_BGR_to_RGB(img_BGR):
    img_RGB = np.zeros_like(img_BGR,dtype=np.float32)
    img_RGB = cv2.cvtColor(src=img_BGR, code=cv2.COLOR_BGR2RGB)
    return img_RGB

def convert_color_space_RGB_to_BGR(img_RGB):
    img_BGR = np.zeros_like(img_RGB,dtype=np.float32)
    img_BGR= cv2.cvtColor(src=img_RGB, code=cv2.COLOR_RGB2BGR)
    return img_BGR

def convert_color_space_RGB_to_Lab(img_RGB_source, img_RGB_target):
    def lab(zx):
        im = zx

        im[0][0]

        base = 10
        Y = np.array([[1., 1., 1.],
                      [1., 1., -2.],
                      [1., -1., 0.]])

        X = np.array([[(1. / pow(3., 0.5)), 0., 0.],
                      [0., (1. / pow(6., 0.5)), 0.],
                      [0., 0., (1. / pow(2., 0.5))]])

        shape = im.shape
        shape
        imlax = np.ones_like(im, dtype=np.float32)
        temp = np.dot(X, Y)
        for i in range(shape[0]):
            for j in range(shape[1]):
                z = im[i][j].reshape(3, 1)
                # if z[0]==0:z[0]=0.00000000001
                # if z[1]==0:z[1]=0.00000000001
                # if z[2]==0:z[2]=0.00000000001
                z = np.log10(z)

                tempp = np.dot(temp, z)
                #         if base == 10:print(tempp)
                temppx = tempp.reshape(1, 3)
                #         if base == 10:print("---------")
                #         if base == 10:print(temppx)
                base = 11
                imlax[i][j] = temppx
        labb = imlax

        l = labb[:, :, 0];
        a = labb[:, :, 1];
        b = labb[:, :, 2];

        lmean = np.mean(l);
        amean = np.mean(a);
        bmean = np.mean(b)

        lm = l - lmean;
        am = a - amean;
        bm = b - bmean;

        lstd = np.std(l);
        astd = np.std(a);
        bstd = np.std(b)
        return (l, a, b, lstd, astd, bstd, lmean, amean, bmean)

    def lms(img_RGB):
        im=img_RGB
        multiplier = np.array([[0.3811, 0.5783, 0.0402], [0.1967, 0.7244, 0.0782], [0.0241, 0.1288, 0.8444]])

        shape = im.shape

        # imlab=np.ones((2,5,3))
        # imlab

        imlab = im
        # imlab

        for i in range(shape[0]):
            for j in range(shape[1]):
                x = im[i][j].reshape(3, 1)

                temp = np.dot(multiplier, x)
                temp = temp.reshape(1, 3)
                imlab[i][j] = temp
        # print(x.shape)
        # print(multiplier.shape)
        # print(temp.shape)

        # print(temp)
        # print(im[0][0])
        # x=im[0][0].reshape(3,1)
        # print(x)
        # ix=np.transpose(x)
        # print(ix)
        img_LMS = np.zeros_like(img_RGB, dtype=np.float32)

        img_LMS = imlab[::]
        labres = lab(img_LMS)
        return labres

    def runlab(img_RGB_source, img_RGB_target):
        s = lms(img_RGB_source)
        t = lms(img_RGB_target)

        newl = (t[3] / s[3]) * (s[0] - s[6])
        newa = (t[4] / s[4]) * (s[1] - s[7])
        newb = (t[5] / s[5]) * (s[2] - s[8])
        l = newl + t[6]
        a = newa + t[7]
        b = newb + t[8]
        # newl = (s[3] / t[3]) * (t[0] - t[6])
        # newa = (s[4] / t[4]) * (t[1] - t[7])
        # newb = (s[5] / t[5]) * (t[2] - t[8])
        # l = newl + s[6]
        # a = newa + s[7]
        # b = newb + s[8]
        img_Lab = np.stack((l, a, b), axis=2)
        return img_Lab
    img_Lab = np.zeros_like(img_RGB_source,dtype=np.float32)
    img_Lab=runlab(img_RGB_source,img_RGB_target)
    # to be completed ...
    return img_Lab

def convert_color_space_Lab_to_RGB(img_Lab):
    targetlab=img_Lab
    def convertlab(targetlab):
        im = targetlab

        im[0][0]

        base = 10
        X = np.array([[1., 1., 1.],
                      [1., 1., -1.],
                      [1., -2., 0.]])

        Y = np.array([[(pow(3., 0.5)/3.), 0., 0.],
                      [0., (pow(6., 0.5)/6.), 0.],
                      [0., 0., (pow(2., 0.5)/2.)]])
        # X = np.array([[1., 1., 1.],
        #               [1., 1., -2.],
        #               [1., -1., 0.]])
        #
        # Y = np.array([[(1 / pow(3, 0.5)), 0., 0.],
        #               [0., (1 / pow(6, 0.5)), 0.],
        #               [0., 0., (1 / pow(2, 0.5))]])
        imlab = im

        shape = im.shape
        shape
        imlax = np.ones_like(im, dtype=np.float32)
        temp = np.dot(X, Y)
        for i in range(shape[0]):
            for j in range(shape[1]):
                z = im[i][j].reshape(3, 1)
                #         if z[0]==0:z[0]=0.0000001
                #         if z[1]==0:z[1]=0.0000001
                #         if z[2]==0:z[2]=0.0000001
                # z= np.log10(z)

                tempp = np.dot(temp, z)
                #         if base == 10:print(tempp)
                temppx = tempp.reshape(1, 3)
                #         if base == 10:print("---------")
                #         if base == 10:print(temppx)
                base = 11
                imlax[i][j] = temppx
        return imlax

    targetlms = convertlab(targetlab)

    def convertlmstorgb(targetlms):
        im = targetlms
        shape = im.shape

        # for i in range(shape[0]):
        #     for j in range(shape[1]):
        #         im[i][j][0]=math.pow(10,im[i][j][0])
        #         im[i][j][1] = math.pow(10, im[i][j][1])
        #         im[i][j][2] = math.pow(10, im[i][j][2])
        im = np.power(10, im)
        return im

    finallms = convertlmstorgb(targetlms)

    def show(finallms):
        im = finallms
        multiplier = np.array([[4.4679, -3.5873, 0.1193], [-1.2186, 2.3809, -0.1624], [0.0497, -0.2439, 1.2045]])

        shape = im.shape

        # imlab=np.ones((2,5,3))
        # imlab

        imlab = im
        # imlab

        for i in range(shape[0]):
            for j in range(shape[1]):
                x = im[i][j].reshape(3, 1)

                temp = np.dot(multiplier, x)
                temp = temp.reshape(1, 3)
                imlab[i][j] = temp

        output = imlab
        return output
        # output = cv2.cvtColor(src=imlab, code=cv2.COLOR_RGB2BGR)
        # imlab=(imlab*255).clip(0.0, 255.0)
        # imlab=imlab.astype(np.uint8)
        # imlab= cv2.cvtColor(src=imlab, code=cv2.COLOR_RGB2BGR)
        # e = np.power((np.mean((resim - output)**2)),0.5)
        # g = max(90*(1 - (e - 0.5) * 0.3), 0)
        # # for i in range (10):
        # #     print(resim[0][i],output[0][i])
        # print(g)
        # def rmse(output, resim):
        #     return np.sqrt(((output - resim) ** 2).mean())
        #
        # e = rmse(np.array(output), np.array(resim))
        # g = max(90*(1 - (e - 0.5) * 0.3), 0)
        # print(g)

        # cv2.imshow('Color image', output)
        # cv2.imwrite("lala.png", imlab)
        #
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    img_RGB=show(finallms)


    return img_RGB


def convert_color_space_RGB_to_CIECAM97s(img_RGB_source, img_RGB_target):

    def lab(zx):
        im = zx

        im[0][0]

        base = 10
        Y = np.array([[2., 1., 0.05],
                      [1., -1.09, 0.09],
                      [0.11, 0.11, -0.22]])



        shape = im.shape
        shape
        imlax = np.ones_like(im, dtype=np.float32)
        temp = Y
        for i in range(shape[0]):
            for j in range(shape[1]):
                z = im[i][j].reshape(3, 1)
                # if z[0]==0:z[0]=0.00000000001
                # if z[1]==0:z[1]=0.00000000001
                # if z[2]==0:z[2]=0.00000000001
                # z = np.log10(z)

                tempp = np.dot(temp, z)
                #         if base == 10:print(tempp)
                temppx = tempp.reshape(1, 3)
                #         if base == 10:print("---------")
                #         if base == 10:print(temppx)
                base = 11
                imlax[i][j] = temppx
        labb = imlax

        l = labb[:, :, 0];
        a = labb[:, :, 1];
        b = labb[:, :, 2];

        lmean = np.mean(l);
        amean = np.mean(a);
        bmean = np.mean(b)

        lm = l - lmean;
        am = a - amean;
        bm = b - bmean;

        lstd = np.std(l);
        astd = np.std(a);
        bstd = np.std(b)
        return (l, a, b, lstd, astd, bstd, lmean, amean, bmean)

    def lms(img_RGB):
        im=img_RGB
        multiplier = np.array([[0.3811, 0.5783, 0.0402], [0.1967, 0.7244, 0.0782], [0.0241, 0.1288, 0.8444]])

        shape = im.shape

        # imlab=np.ones((2,5,3))
        # imlab

        imlab = im
        # imlab

        for i in range(shape[0]):
            for j in range(shape[1]):
                x = im[i][j].reshape(3, 1)

                temp = np.dot(multiplier, x)
                temp = temp.reshape(1, 3)
                imlab[i][j] = temp
        # print(x.shape)
        # print(multiplier.shape)
        # print(temp.shape)

        # print(temp)
        # print(im[0][0])
        # x=im[0][0].reshape(3,1)
        # print(x)
        # ix=np.transpose(x)
        # print(ix)
        img_LMS = np.zeros_like(img_RGB, dtype=np.float32)

        img_LMS = imlab[::]
        labres = lab(img_LMS)
        return labres

    def runlab(img_RGB_source, img_RGB_target):
        s = lms(img_RGB_source)
        t = lms(img_RGB_target)

        newl = (t[3] / s[3]) * (s[0] - s[6])
        newa = (t[4] / s[4]) * (s[1] - s[7])
        newb = (t[5] / s[5]) * (s[2] - s[8])
        l = newl + t[6]
        a = newa + t[7]
        b = newb + t[8]
        # newl = (t[3] / s[3]) * (s[0] - s[6])
        # newa = (t[4] / s[4]) * (s[1] - s[7])
        # newb = (t[5] / s[5]) * (s[2] - s[8])
        # l = newl + t[6]
        # a = newa + t[7]
        # b = newb + t[8]

        img_Lab = np.stack((l, a, b), axis=2)
        return img_Lab
    img_Lab = np.zeros_like(img_RGB_source,dtype=np.float32)
    img_Lab=runlab(img_RGB_source,img_RGB_target)
    # to be completed ...
    return img_Lab

def convert_color_space_CIECAM97s_to_RGB(img_CIECAM97s):
    targetlab=img_CIECAM97s
    def convertlab(targetlab):
        im = targetlab

        im[0][0]

        base = 10
        Y = np.array([[2., 1., 0.05],
                      [1., -1.09, 0.09],
                      [0.11, 0.11, -0.22]])
        Y=np.linalg.inv(Y)
        imlab = im

        shape = im.shape
        shape
        imlax = np.ones_like(im, dtype=np.float32)
        temp = Y
        for i in range(shape[0]):
            for j in range(shape[1]):
                z = im[i][j].reshape(3, 1)
                #         if z[0]==0:z[0]=0.0000001
                #         if z[1]==0:z[1]=0.0000001
                #         if z[2]==0:z[2]=0.0000001
                #         z= np.log10(z)

                tempp = np.dot(temp, z)
                #         if base == 10:print(tempp)
                temppx = tempp.reshape(1, 3)
                #         if base == 10:print("---------")
                #         if base == 10:print(temppx)
                base = 11
                imlax[i][j] = temppx
        return imlax

    targetlms = convertlab(targetlab)

    def convertlmstorgb(targetlms):
        im = targetlms
        shape = im.shape

        # for i in range(shape[0]):
        #     for j in range(shape[1]):
        #         im[i][j][0]=math.pow(10,im[i][j][0])
        #         im[i][j][1] = math.pow(10, im[i][j][1])
        #         im[i][j][2] = math.pow(10, im[i][j][2])
        # im = np.power(10, im)
        return im

    finallms = convertlmstorgb(targetlms)

    def show(finallms):
        im = finallms
        multiplier = np.array([[4.4679, -3.5873, 0.1193], [-1.2186, 2.3809, -0.1624], [0.0497, -0.2439, 1.2045]])

        shape = im.shape

        # imlab=np.ones((2,5,3))
        # imlab

        imlab = im
        # imlab

        for i in range(shape[0]):
            for j in range(shape[1]):
                x = im[i][j].reshape(3, 1)

                temp = np.dot(multiplier, x)
                temp = temp.reshape(1, 3)
                imlab[i][j] = temp

        output = imlab
        return output
        # output = cv2.cvtColor(src=imlab, code=cv2.COLOR_RGB2BGR)
        # imlab=(imlab*255).clip(0.0, 255.0)
        # imlab=imlab.astype(np.uint8)
        # imlab= cv2.cvtColor(src=imlab, code=cv2.COLOR_RGB2BGR)
        # e = np.power((np.mean((resim - output)**2)),0.5)
        # g = max(90*(1 - (e - 0.5) * 0.3), 0)
        # # for i in range (10):
        # #     print(resim[0][i],output[0][i])
        # print(g)
        # def rmse(output, resim):
        #     return np.sqrt(((output - resim) ** 2).mean())
        #
        # e = rmse(np.array(output), np.array(resim))
        # g = max(90*(1 - (e - 0.5) * 0.3), 0)
        # print(g)

        # cv2.imshow('Color image', output)
        # cv2.imwrite("lala.png", imlab)
        #
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    img_RGB=show(finallms)


    return img_RGB

def convert_color_space_RGB_to_RGBs(img_RGB_source, img_RGB_target):
    def lab(zx):
        im = zx

        im[0][0]

        base = 10
        Y = np.array([[1., 1., 1.],
                      [1., 1., 1.],
                      [1., 1., 1.]])



        shape = im.shape
        shape
        imlax = np.ones_like(im, dtype=np.float32)
        temp = Y
        for i in range(shape[0]):
            for j in range(shape[1]):
                z = im[i][j].reshape(3, 1)
                # if z[0]==0:z[0]=0.00000000001
                # if z[1]==0:z[1]=0.00000000001
                # if z[2]==0:z[2]=0.00000000001
                # z = np.log10(z)

                tempp = np.dot(temp, z)
                #         if base == 10:print(tempp)
                temppx = tempp.reshape(1, 3)
                #         if base == 10:print("---------")
                #         if base == 10:print(temppx)
                base = 11
                imlax[i][j] = temppx
        labb = im

        l = labb[:, :, 0];
        a = labb[:, :, 1];
        b = labb[:, :, 2];

        lmean = np.mean(l);
        amean = np.mean(a);
        bmean = np.mean(b)

        lm = l - lmean;
        am = a - amean;
        bm = b - bmean;

        lstd = np.std(l);
        astd = np.std(a);
        bstd = np.std(b)
        return (l, a, b, lstd, astd, bstd, lmean, amean, bmean)

    def lms(img_RGB):
        im=img_RGB
        multiplier = np.array([[0.3811, 0.5783, 0.0402], [0.1967, 0.7244, 0.0782], [0.0241, 0.1288, 0.8444]])

        shape = im.shape

        # imlab=np.ones((2,5,3))
        # imlab

        imlab = im
        # imlab

        for i in range(shape[0]):
            for j in range(shape[1]):
                x = im[i][j].reshape(3, 1)

                temp = np.dot(multiplier, x)
                temp = temp.reshape(1, 3)
                imlab[i][j] = temp
        # print(x.shape)
        # print(multiplier.shape)
        # print(temp.shape)

        # print(temp)
        # print(im[0][0])
        # x=im[0][0].reshape(3,1)
        # print(x)
        # ix=np.transpose(x)
        # print(ix)
        img_LMS = np.zeros_like(img_RGB, dtype=np.float32)

        img_LMS = imlab[::]
        labres = lab(img_LMS)
        return labres

    def runlab(img_RGB_source, img_RGB_target):
        s = lms(img_RGB_source)
        t = lms(img_RGB_target)

        newl = (t[3] / s[3]) * (s[0] - s[6])
        newa = (t[4] / s[4]) * (s[1] - s[7])
        newb = (t[5] / s[5]) * (s[2] - s[8])
        l = newl + t[6]
        a = newa + t[7]
        b = newb + t[8]

        img_Lab = np.stack((l, a, b), axis=2)
        return img_Lab
    img_Lab = np.zeros_like(img_RGB_source,dtype=np.float32)
    img_Lab=runlab(img_RGB_source,img_RGB_target)
    # to be completed ...
    return img_Lab

def convert_color_space_RGBs_to_RGB(img_CIECAM97s):
    targetlab=img_CIECAM97s
    def convertlab(targetlab):
        im = targetlab

        im[0][0]

        base = 10
        Y = np.array([[1.00001, 1.00001, 1.00001],
                      [1.00001, 1.00001, 1.00001],
                      [1.00001, 1.00001, 1.00001]])


        imlab = im

        shape = im.shape
        shape
        imlax = np.ones_like(im, dtype=np.float32)
        temp = Y
        for i in range(shape[0]):
            for j in range(shape[1]):
                z = im[i][j].reshape(3, 1)
                #         if z[0]==0:z[0]=0.0000001
                #         if z[1]==0:z[1]=0.0000001
                #         if z[2]==0:z[2]=0.0000001
                #         z= np.log10(z)

                tempp = np.dot(temp, z)
                #         if base == 10:print(tempp)
                temppx = tempp.reshape(1, 3)
                #         if base == 10:print("---------")
                #         if base == 10:print(temppx)
                base = 11
                imlax[i][j] = temppx
        return im

    targetlms = convertlab(targetlab)

    def convertlmstorgb(targetlms):
        im = targetlms
        shape = im.shape

        # for i in range(shape[0]):
        #     for j in range(shape[1]):
        #         im[i][j][0]=math.pow(10,im[i][j][0])
        #         im[i][j][1] = math.pow(10, im[i][j][1])
        #         im[i][j][2] = math.pow(10, im[i][j][2])
        # im = np.power(10, im)
        return im

    finallms = convertlmstorgb(targetlms)

    def show(finallms):
        im = finallms
        multiplier = np.array([[4.4679, -3.5873, 0.1193], [-1.2186, 2.3809, -0.1624], [0.0497, -0.2439, 1.2045]])

        shape = im.shape

        # imlab=np.ones((2,5,3))
        # imlab

        imlab = im
        # imlab

        for i in range(shape[0]):
            for j in range(shape[1]):
                x = im[i][j].reshape(3, 1)

                temp = np.dot(multiplier, x)
                temp = temp.reshape(1, 3)
                imlab[i][j] = temp

        output = imlab
        return output
        # output = cv2.cvtColor(src=imlab, code=cv2.COLOR_RGB2BGR)
        # imlab=(imlab*255).clip(0.0, 255.0)
        # imlab=imlab.astype(np.uint8)
        # imlab= cv2.cvtColor(src=imlab, code=cv2.COLOR_RGB2BGR)
        # e = np.power((np.mean((resim - output)**2)),0.5)
        # g = max(90*(1 - (e - 0.5) * 0.3), 0)
        # # for i in range (10):
        # #     print(resim[0][i],output[0][i])
        # print(g)
        # def rmse(output, resim):
        #     return np.sqrt(((output - resim) ** 2).mean())
        #
        # e = rmse(np.array(output), np.array(resim))
        # g = max(90*(1 - (e - 0.5) * 0.3), 0)
        # print(g)

        # cv2.imshow('Color image', output)
        # cv2.imwrite("lala.png", imlab)
        #
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    img_RGB=show(finallms)


    return img_RGB


def color_transfer_in_Lab(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_Lab =====')
    img_Lab=convert_color_space_RGB_to_Lab(img_RGB_source, img_RGB_target)
    Rgb=convert_color_space_Lab_to_RGB(img_Lab)


    return Rgb
    img_Lab = convert_color_space_RGB_to_Lab(img_RGB_source, img_RGB_target)
    Rgb = convert_color_space_Lab_to_RGB(img_Lab)

    return Rgb



def color_transfer_in_RGB(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_RGB =====')
    img_Lab = convert_color_space_RGB_to_RGBs(img_RGB_source, img_RGB_target)
    Rgb = convert_color_space_RGBs_to_RGB(img_Lab)
    return Rgb


def color_transfer_in_CIECAM97s(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_CIECAM97s =====')
    img_Lab = convert_color_space_RGB_to_CIECAM97s(img_RGB_source, img_RGB_target)
    Rgb = convert_color_space_CIECAM97s_to_RGB(img_Lab)

    return Rgb


def color_transfer(img_RGB_source, img_RGB_target, option):
    if option == 'in_RGB':
        img_RGB_new = color_transfer_in_RGB(img_RGB_source, img_RGB_target)
    elif option == 'in_Lab':
        img_RGB_new = color_transfer_in_Lab(img_RGB_source, img_RGB_target)
    elif option == 'in_CIECAM97s':
        img_RGB_new = color_transfer_in_CIECAM97s(img_RGB_source, img_RGB_target)
    return img_RGB_new

if __name__ == "__main__":
    print('==================================================')
    print('PSU CS 410/510, Winter 2019, HW1: color transfer')
    print('==================================================')

    # path_file_image_source = "source.png"
    # path_file_image_target = "target.png"
    # path_file_image_result_in_Lab = "reslab.png"
    # path_file_image_result_in_RGB = "resrgb.png"
    # path_file_image_result_in_CIECAM97s = "rescie.png"
    path_file_image_source = sys.argv[1]
    path_file_image_target = sys.argv[2]
    path_file_image_result_in_Lab = sys.argv[3]
    path_file_image_result_in_RGB = sys.argv[4]
    path_file_image_result_in_CIECAM97s = sys.argv[5]


    # ===== read input images
    im= cv2.imread(path_file_image_source , flags=cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    img_RGB_source=convert_color_space_BGR_to_RGB(im)
    imr=cv2.imread(path_file_image_target, flags=cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    img_RGB_target= convert_color_space_BGR_to_RGB(imr)

    img_RGB_new_Lab= color_transfer(img_RGB_source, img_RGB_target, option='in_Lab')

    output=convert_color_space_RGB_to_BGR(img_RGB_new_Lab)


    # cv2.imshow('output image',output)
    cv2.imwrite(path_file_image_result_in_Lab,img=(output * 255.0).clip(0.0, 255.0).astype(np.uint8))



    #
    # #
    # #
    # #
    # #
    # img_RGB_new_RGB       = color_transfer(img_RGB_source, img_RGB_target, option='in_RGB')
    # # # # todo: save image to path_file_image_result_in_RGB
    # #
    img_RGB_new_CIECAM97s = color_transfer(img_RGB_source, img_RGB_target, option='in_CIECAM97s')
    output=convert_color_space_RGB_to_BGR(img_RGB_new_CIECAM97s)

    # cv2.imshow('ciecam image',output)

    cv2.imwrite(path_file_image_result_in_CIECAM97s, img=(output * 255.0).clip(0.0, 255.0).astype(np.uint8))
    #
    # #
    # #
    img_RGB_new_RGB= color_transfer(img_RGB_source, img_RGB_target, option='in_RGB')
    # # todo: save image to path_file_image_result_in_RGB
    output=convert_color_space_RGB_to_BGR(img_RGB_new_RGB)


    # cv2.imshow('rgb image',output)

    cv2.imwrite(path_file_image_result_in_RGB, img=(output * 255.0).clip(0.0, 255.0).astype(np.uint8))

    cv2.waitKey(0)

    resim = cv2.imread("result2.png", cv2.IMREAD_COLOR).astype(np.float32)
    output = cv2.imread("reslab2.png", cv2.IMREAD_COLOR).astype(np.float32)
    # def rmse(output, resim):
    #     return np.sqrt(((output - resim) ** 2).mean())
    #
    #
    # #
    # e = rmse(np.array(output), np.array(resim))
    # g = max(90 * (1 - (e - 0.5) * 0.3), 0)
    # print(resim[0][0], output[0][0],e)