from tqdm import tqdm
import torch 
from utils import get_confusion_matrix, get_precision_recall_f1
import numpy as np


def train_one_epoch(model, dataloder, optimizer, criterion, epoch, args):
    
    model.train()

    bar = tqdm(dataloder, desc=f'Epoch {epoch}')
    total_loss = 0
    count = 0

    for images, labels in bar:
        images = images.to(args.device)
        labels = labels.to(args.device)

        count += 1
        output = model(images)

        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        bar.set_postfix_str(f'Batch loss: {loss.item()} Total Loss: {total_loss / count}', refresh=True)
        bar.update()
    
    total_loss /= count

    return total_loss


@torch.no_grad()
def evaluate(model, dataloder, criterion, args):
    
    model.eval()
    
    bar = tqdm(dataloder, desc=f'Test')
    total_loss = 0
    count = 0
    
    all_predictions = torch.tensor([], device=args.device, dtype=int)
    all_labels = torch.tensor([], device=args.device, dtype=int)
    correct_prediction = 0

    for images, labels in bar:
        images = images.to(args.device)
        labels = labels.to(args.device)

        count += 1
        output = model(images)
        
        _, prediction = output.max(dim = 1)
        
        all_predictions = torch.cat((all_predictions, prediction), 0)
        all_labels = torch.cat((all_labels, labels), 0)
        

        correct_prediction += (prediction == labels).sum()
        loss = criterion(output, labels)
        total_loss += loss.item()

        bar.set_postfix_str(f'Batch loss: {loss.item()} Total Loss: {total_loss / count}')
        bar.update()
    
    total_loss /= count
    
    accuracy = ((correct_prediction * 100) / (count * args.batch_size)).item()


    cm = get_confusion_matrix(all_predictions.cpu(), all_labels.cpu(), args.num_classes)
    precision, recall, f1_score = get_precision_recall_f1(cm)
    

    return total_loss, accuracy, precision, recall, f1_score
    
    




