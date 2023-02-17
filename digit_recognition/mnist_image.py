import torch
import torchvision 
from PIL import Image 
import numpy


train_data = torchvision.datasets.MNIST('./data',
                            train = True,
                            download = True,
                            transform = torchvision.transforms.ToTensor())

test_data = torchvision.datasets.MNIST('./data',
                            train=False,
                            download=True,
                            transform = torchvision.transforms.ToTensor())


def main():
    print(len(train_data), len(test_data))

    img, label = train_data[1]
    print("*****************************")    
    print(img)

    np_img = img.numpy()
    
    print("*****************************")    
    print(np_img)

    re_img = np_img.reshape(28, 28) 
   
    print("*****************************")    
    print(re_img)
    
    ary = numpy.uint8(numpy.around(re_img))
    #show 
    print(ary)
    print("*****************************") 

    print("label is: ",label)
    
    #normalization to bmp
    ary *= 255;
    
    #convert array to image
    pil_img = Image.fromarray(ary, mode='L')
    pil_img.save("./tmp.png");

if __name__ == '__main__':
    main()


