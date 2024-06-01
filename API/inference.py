import torch
from torchvision import transforms
from model import DocumentOrientationModel


def inference(image):
    best_model = 'best_model.pth'
    model = DocumentOrientationModel(num_classes=8)
    
    checkpoint = torch.load(best_model, map_location='cpu')
    model.load_state_dict(checkpoint)

    model.eval()
    


    transform  = transforms.Compose([
        transforms.RandomResizedCrop((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    image = transform(image)
    image = image
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

