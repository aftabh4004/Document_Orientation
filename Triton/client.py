from PIL import Image
from torchvision import transforms
import tritonclient.http as httpclient
import sys
from pdf2image import convert_from_path

def preprocess(image):

    image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.RandomResizedCrop((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image)
    return image.numpy()


argv = sys.argv


if len(argv) != 2:
    print("Please provide file")
    exit(0)

image_path = argv[1]


if image_path.endswith('jpg') or image_path.endswith('jpeg') or image_path.endswith('png'):
    image = Image.open(image_path)
    npimage = preprocess(image)
elif image_path.endswith('pdf'):
    image = convert_from_path(image_path)[0]
    npimage = preprocess(image)
else:
    print("Invalid file format")
    exit(0)
 


client = httpclient.InferenceServerClient(url='localhost:8000')

inputs = httpclient.InferInput("input__0", npimage.shape, datatype='FP32')
outputs  = httpclient.InferRequestedOutput('output__0', binary_data=False, class_count=8)


inputs.set_data_from_numpy(npimage, binary_data=True)


#query

prediction = client.infer(model_name='doc-orientation', inputs=[inputs], outputs=[outputs])
prediction = prediction.as_numpy('output__0')

cls = int(prediction[0].split(':')[-1])
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

print(f'{360 - (cls * 45 + 45) % 360} degree clockwise' )

