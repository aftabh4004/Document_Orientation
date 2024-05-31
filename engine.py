from tqdm import tqdm
import torch 
from utils import get_confusion_matrix, get_precision_recall_f1
import numpy as np


def train_one_epoch(model, dataloder, optimizer, criterion, epoch):
    
    model.train()

    bar = tqdm(dataloder, desc=f'Epoch {epoch}')
    total_loss = 0
    count = 0

    for images, labels in bar:
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
def evaluate(model, dataloder, criterion, batch_size, num_classes):
    model.eval()
    bar = tqdm(dataloder, desc=f'Test')
    total_loss = 0
    count = 0
    all_predictions = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)
    correct_prediction = 0

    for images, labels in bar:
        count += 1
        output = model(images)
        
        _, prediction = output.max(dim = 1)
        
        all_predictions = np.concatenate((all_predictions, prediction), axis = None)
        all_labels = np.concatenate((all_labels, labels), axis = None)
        

        correct_prediction += (prediction == labels).sum()
        loss = criterion(output, labels)
        total_loss += loss.item()

        bar.set_postfix_str(f'Batch loss: {loss.item()} Total Loss: {total_loss / count}')
        bar.update()
    
    total_loss /= count
    
    accuracy = ((correct_prediction * 100) / (count * batch_size)).item()


    cm = get_confusion_matrix(all_predictions, all_labels, num_classes)
    precision, recall, f1_score = get_precision_recall_f1(cm)
    

    return total_loss, accuracy, precision, recall, f1_score
    
    




