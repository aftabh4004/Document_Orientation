import torch
from torchvision import transforms
from PIL import Image

best_model = 'output/img_size448_ep100bs32lr0.0003/best_model.pth'

def inference(image):
    model = torch.load(best_model)
    model.eval()
    model = model.to("cuda")


    transform  = transforms.Compose([
        transforms.RandomResizedCrop((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    image = transform(image)
    image = image.to("cuda")
    image = image.unsqueeze(dim=0)
    
    
    output = model(image)
    _, prediction = output.max(dim=1)
    prediction = prediction[0].item()

    # Convert to degree
    # degree    label
    # 45          6
    # 90          5
    # 135         4
    # 180         3
    # 225         2
    # 270         1
    # 315         0
    # 360         7
    
    return 360 - (prediction * 45 + 45) % 360

image_path = 'dataset/active/3418_68_225.jpg'
image = Image.open(image_path)

angle = inference(image=image)

print(angle)

