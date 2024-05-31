from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch



class Record:
    def __init__(self, path, rotation):
        self.path = path
        self.rotation = rotation

class DocumentOrientationDataset(Dataset):
    def __init__(self, dataset_list_file: str, transform: transforms):
        self.records = []
        self.transform = transform
        self._parse_list(dataset_list_file)

    
    def _parse_list(self, path):
        with open(path, 'r') as fp:
            lines = fp.readlines()

            for line in lines:
                img_path, rotation = line.split('\t')
                self.records += [Record(img_path, int(rotation))]
    

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]

        image = Image.open(record.path)
        label = record.rotation
       

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        

        return image, label

