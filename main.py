import argparse
import os
from dataset import DocumentOrientationDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from model import DocumentOrientationModel
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from engine import train_one_epoch, evaluate
import torch

def get_args_parser():
    parser = argparse.ArgumentParser('Documents orientation classification training')
    parser.add_argument('--output-dir', type=str, default='output/default')
    parser.add_argument('--dataset-list-root', type=str)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=8)
    
    
    return parser

def main(args):
    
    train_list = os.path.join(args.dataset_list_root, 'train.txt')
    test_list = os.path.join(args.dataset_list_root, 'test.txt')

    transform = transforms.Compose([   
        transforms.RandomResizedCrop((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_train = DocumentOrientationDataset(train_list, transform)
    dataset_test = DocumentOrientationDataset(test_list, transform)


    dataloder_train = DataLoader(dataset_train, batch_size = args.batch_size, shuffle=True)
    dataloder_test = DataLoader(dataset_test, batch_size = args.batch_size, shuffle=True)

    
    model = DocumentOrientationModel(num_classes=args.num_classes)
    
    optimizer = Adam(lr = args.lr, params=model.parameters())
    criterion = CrossEntropyLoss()

    logfile = os.path.join(args.output_dir, 'log.txt')
    with open(logfile, 'w') as fp:
        print(f'{"Epoch":10}{"train_loss":15}{"test_loss":15}{"precision":15}{"recall":15}{"f1_score":15}{"accracy":15}', file=fp)
    
    max_acc = 0

    for epoch in range(args.epochs):
        
        train_loss = train_one_epoch(model, dataloder_train, optimizer, criterion, epoch)
        test_loss, accuracy, precision, recall, f1_score = evaluate(model, dataloder_test, criterion, args.batch_size, args.num_classes)

        logstr = f'{epoch:<10}{train_loss:<15.3f}{test_loss:<15.3f}{precision:<15.3f}{recall:<15.3f}{f1_score:<15.3f}{accuracy:<15.3f}'

        with open(logfile, 'a') as fp:
            print(logstr, file=fp)

        # Save best model
        max_acc = max(max_acc, accuracy)

        if accuracy == max_acc:
            torch.save(model, os.path.join(args.output_dir, 'best_model.pth'))    

    
if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok = True)
    main(args)

