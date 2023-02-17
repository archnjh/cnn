from PIL import Image
from LeNet5 import LeNet5
import torch
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trans = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])

def predict(image):
    net = LeNet5()
    net.load_state_dict(torch.load('./params.pkl'))
    net.to(device)

    output = net(image)

    print("output:" + str(output))
    print("output shape:" + str(output.shape))
    print("output size:" + str(output.size()))

    ret_val, predicted = torch.max(output, 1)
    _, predicted = torch.max(output, 1)

    print("ret_val:" + str(ret_val))
    print("predicted:" + str(predicted))

    return predicted

def recognition():
    print("begin to recogition")
    test_img = Image.open('./test.png')
    resize_img = test_img.resize((28,28))
    gray_img = resize_img.convert('L')

    trans_img = trans(gray_img).unsqueeze(0)
    input_img = trans_img.to(device)

    result = predict(input_img)

    print(result)

    print("result is :" + str(result.item()))

if __name__ == "__main__":
    recognition()
    