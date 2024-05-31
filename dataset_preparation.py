import os
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms 

image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
dataset_list_train = './dataset_list/train.txt'
dataset_list_test = './dataset_list/test.txt'

images = []
rotation = []
directory = 'dataset/original'
output_dir = 'dataset/active'
counter = 0

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.lower().endswith(image_extensions):
            image_path = os.path.join(root, file)
            try:
                with Image.open(image_path) as img:
                    
                    img = img.convert('RGB')
                    img_t = transforms.ToTensor()(img)
                    

                    assert img.mode == 'RGB' , f'{image_path} no RGB mode'
                    assert img_t.shape[0] == 3, f'{image_path}, shape {img_t.shape}'

                    angles = [45, 90, 135, 180, 225, 270, 315, 360]
                    for i, angle in enumerate(angles):
                        rotated_image = img.rotate(angle, expand=True)

                        image_name = image_path.split('/')[-1].split('.')[0]
                        image_ext = image_path.split('.')[-1]

                        output_path = os.path.join(output_dir, f'{counter}_{image_name}_{360-angle}.{image_ext}')
                        rotated_image.save(output_path)
                        counter += 1
                        
                        images.append(output_path)
                        rotation.append(i)
            except Exception as e:
                print(f'Error in {image_path}')
                print(e)




X_train, X_test, y_train, y_test = train_test_split(
    images, rotation, test_size=0.2, random_state=42, stratify=rotation
)

# Verify the class distribution
from collections import Counter
print(f"Training set class distribution: {Counter(y_train)}")
print(f"Test set class distribution: {Counter(y_test)}")


with open(dataset_list_train, 'w') as fp:
    for i in range(len(X_train)):
        print(f'{X_train[i]}\t{y_train[i]}', file=fp)

with open(dataset_list_test, 'w') as fp:
    for i in range(len(X_test)):
        print(f'{X_test[i]}\t{y_test[i]}', file=fp)

