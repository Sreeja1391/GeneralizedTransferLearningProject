import time
import random
import torch.nn.functional as F
import numpy as np
import torch
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision.transforms import InterpolationMode
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt
from pathlib import Path
import opendatasets as od
import os
import shutil
from sklearn.metrics import classification_report

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prepare_data(args, flag):
    print("Loading data...", end="")
    if flag == 'train':
        train_transform = transforms.Compose([transforms.RandomRotation(degrees=5, interpolation=InterpolationMode.BILINEAR),
                                              transforms.RandomVerticalFlip(p=0.1),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=(0.5,), std=(0.5,))])
        if args.dataset == 'CIFAR10':
            dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        elif args.dataset == 'Fruits360':
            if not os.path.exists('./data/Fruits360'):
                od.download('https://github.com/Horea94/Fruit-Images-Dataset/archive/refs/heads/master.zip')
                shutil.unpack_archive('Fruit-Images-Dataset-master.zip', './data')
                os.rename('./data/Fruit-Images-Dataset-master', './data/Fruits360')
            dataset = ImageFolder('./data/Fruits360/Training', transform=train_transform)
        valid_size = int(len(dataset) * args.train_valid_split)
        train_size = len(dataset) - valid_size
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        print('train_size:', len(train_dataset), 'valid_size:', len(valid_dataset))
        return train_loader, valid_loader, len(dataset.classes)
    elif flag == 'test':
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.5,), std=(0.5,))])

        if args.dataset == 'CIFAR10':
            test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        elif args.dataset == 'Fruits360':
            test_dataset = ImageFolder('./data/Fruits360/Test', transform=test_transform)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        print('test_size:', len(test_dataset))
        return test_loader, test_dataset, len(test_dataset.classes)


def compute_accuracy_and_loss(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    cross_entropy = 0.
    for i, (features, targets) in enumerate(tqdm(data_loader)):
        features = features.to(device)
        targets = targets.to(device)

        logits = model(features)
        cross_entropy += F.cross_entropy(logits, targets).item()
        _, preds = torch.max(logits, dim=1)
        num_examples += targets.size(0)
        correct_pred += (preds == targets).sum()
    return correct_pred.float()/num_examples, cross_entropy/num_examples


def save_model(model, optimizer, args):
    print("Saving model...", end="")
    info = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args,
        'system_random': random.getstate(),
        'numpy_random': np.random.get_state(),
        'torch_random': torch.random.get_rng_state(),
    }
    torch.save(info, args.filepath)
    print(f"to {args.filepath}")


def save_loss(epoch, train_acc, train_loss, valid_acc, valid_loss):
    print("Saving loss...", end="")
    losspath = Path(args.filepath).stem + '-loss.pt'
    info = {
        'epoch': epoch,
        'train_acc': train_acc,
        'train_loss': train_loss,
        'valid_acc': valid_acc,
        'valid_loss': valid_loss,
    }
    torch.save(info, losspath)
    print(f"to {losspath}")


def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    train_loader, valid_loader, num_class = prepare_data(args, 'train')

    if args.model_name == 'resnet18':
        model = models.resnet18(weights=args.model_weight)
    elif args.model_name == 'vgg16':
        model = models.vgg16(weights=args.model_weight)

    for param in model.parameters():    # freeze all layers except last layer
        param.requires_grad = False

    if args.model_name == 'resnet18':
        # model.fc = torch.nn.Linear(model.fc.in_features, num_class)
        model.fc.out_features = num_class
    elif args.model_name == 'vgg16':
        model.classifier[6].out_features = num_class

    for ind_layer, param in enumerate(model.parameters()):
        if ind_layer < args.freeze_layers:  # freeze all layers from 1st to {args.freeze_layers}th layer
            param.requires_grad = False
        else:
            param.requires_grad = True

    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name, layer.trainable)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_time = time.time()
    train_acc = torch.zeros(args.epochs, dtype=torch.float)
    train_loss = torch.zeros(args.epochs, dtype=torch.float)
    valid_acc = torch.zeros(args.epochs, dtype=torch.float)
    valid_loss = torch.zeros(args.epochs, dtype=torch.float)
    best_valid_acc = 0

    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (features, targets) in enumerate(tqdm(train_loader)):
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = F.cross_entropy(logits, targets)

            loss.backward()
            optimizer.step()

            if not batch_idx % 100:
                print(f'epoch: {epoch+1:02d}/{args.epochs:02d} | '
                      f'batch {batch_idx:03d}/{len(train_loader):03d} |' 
                      f' loss: {loss:.4f}')

        model.eval()
        with torch.set_grad_enabled(False):
            train_acc[epoch], train_loss[epoch] = compute_accuracy_and_loss(model, train_loader, device=device)
            valid_acc[epoch], valid_loss[epoch] = compute_accuracy_and_loss(model, valid_loader, device=device)
            print(f'epoch: {epoch+1:02d}/{args.epochs:02d} train acc: {train_acc[epoch]:.4f} valid acc: {valid_acc[epoch]:.4f}')
            save_loss(epoch, train_acc, train_loss, valid_acc, valid_loss)

        if valid_acc[epoch] > best_valid_acc:
            best_valid_acc = valid_acc[epoch]
            save_model(model, optimizer, args)
        
        elapsed = (time.time() - start_time)/60
        print(f'Time elapsed: {elapsed:.2f} min')

    elapsed = (time.time() - start_time)/60
    print(f'Total Training Time: {elapsed:.2f} min')

    
'''
3 functions defined to simplify: 
get_classification_report, get_true_and_pred_labels and check_accuracy
'''

def get_classification_report(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names)
    return report

def get_true_and_pred_labels(model, data_loader, device):
    y_true = []
    y_pred = []
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
    return y_true, y_pred

def check_accuracy(test_loader, loader_dataset, num_classes, model):
    
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    labels_map = loader_dataset.classes
    
    running_corrects = 0
    num_samples = 0
    n_correct_class = [0 for i in range(num_classes)]
    n_class_samples = [0 for i in range(num_classes)]
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            y_predicted = model(images)
            _, index = torch.max(y_predicted, 1)
            running_corrects += (index == labels.data).sum()
            temp_ = index.cpu().numpy()
            num_samples += temp_.shape[0]

            temp = labels.cpu().numpy()
            for i in range(temp.shape[0]):
                label = temp[i]
                index_i = temp_[i]

                if label == index_i:
                    n_correct_class[label] += 1
                n_class_samples[label] += 1

        convert = running_corrects.double()
        acc = convert / len(loader_dataset)
        print(f'Got {int(convert.item())}/{num_samples} correct samples over {acc.item() * 100:.2f}%')

        for i in range(num_classes):
            if n_class_samples[i] != 0:
                acc_ = 100 * n_correct_class[i] / n_class_samples[i]
                print(f'Accuracy of {labels_map[i]}: {acc_:.2f}%')
            else:
                print(f'Class {labels_map[i]} does not have its sample in this dataset.')
                

def visualize_model(num_rows, num_cols, dataset, predicted_class):
    save_path = 'my_image.png'
    mean = torch.tensor([0.5,]) 
    std = torch.tensor([0.5,])
    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    labels_map = dataset.classes
    
    for i in range(num_rows*num_cols):
        ax = fig.add_subplot(num_rows, num_cols, i+1)
        #plt.subplot(num_rows, num_cols, i+1)
        index = torch.randint(len(dataset), size=(1,)).item()
        image, label = dataset[index]
        image = image.cpu().numpy().transpose(1,2,0)
        image = image * np.array(std) + np.array(mean)
        
        ax_image = ax.imshow(image)
        ax.axis('off')
        if torch.tensor(label).item() == predicted_class[index]:
            check = 'green'
        else: 
            check = 'red'
        ax.set_title(f'Pred: {labels_map[predicted_class[index]]}', color='white', backgroundcolor=check, fontsize=15)
        
    fig.savefig(save_path)
                    
def test(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    test_loader, test_dataset, num_class = prepare_data(args, 'test')

    print("Loading model...", end="")
    checkpoint = torch.load(args.filepath)

    # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    if args.model_name == 'resnet18':
        model = models.resnet18(weights=None)   # random initialization
        model.fc.out_features = num_class
        # model.fc = torch.nn.Linear(model.fc.in_features, num_class)
    elif args.model_name == 'vgg16':
        model = models.vgg16(weights=None)  # random initialization
        model.classifier[6].out_features = num_class

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"from {args.filepath}")

    model.eval()
    with torch.set_grad_enabled(False):     # save memory during inference
        test_acc, test_loss = compute_accuracy_and_loss(model, test_loader, device)
        print(f'test acc: {test_acc:.4f}')
        '''
        CHANGES HERE:
        ''' 
        y_true, y_pred = get_true_and_pred_labels(model, test_loader, device)
        class_names = [f'class {i}' for i in range(num_class)]
        report = get_classification_report(y_true, y_pred, class_names)
        print(report)
        check_accuracy(test_loader, test_dataset, num_class, model)
        visualize_model(2,2,test_dataset, y_pred)
        
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    
    if args.create_plot:
        losspath = Path(args.filepath).stem + '-loss.pt'
        plotpath = Path(args.filepath).stem

        print("Loading loss...", end="")
        checkpoint = torch.load(losspath)
        print(f"from {args.filepath}")

        train_loss = checkpoint['train_loss'].detach().cpu().numpy()
        valid_loss = checkpoint['valid_loss'].detach().cpu().numpy()
        plt.figure()
        plt.plot(range(1, args.epochs + 1), train_loss, label='train loss')
        plt.plot(range(1, args.epochs + 1), valid_loss, label='valid loss')
        plt.legend(loc='upper right')
        plt.ylabel('cross entropy loss')
        plt.xlabel('epoch')
        plt.savefig(plotpath + '-loss.png')

        train_acc = checkpoint['train_acc'].detach().cpu().numpy()
        valid_acc = checkpoint['valid_acc'].detach().cpu().numpy()
        plt.figure()
        plt.plot(range(1, args.epochs + 1), train_acc, label='train accuracy')
        plt.plot(range(1, args.epochs + 1), valid_acc, label='valid accuracy')
        plt.legend(loc='upper left')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig(plotpath + '-acc.png')


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=('CIFAR10', 'Fruits360'))
    parser.add_argument("--model_name", type=str, default="resnet18", choices=('resnet18', 'vgg16'))
    parser.add_argument("--model_weight", type=str, default="IMAGENET1K_V1")
    parser.add_argument("--freeze_layers", type=int, default=17)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--train_valid_split", type=float, default=0.1)
    parser.add_argument("--filepath", type=str, default=".pt")
    parser.add_argument("--create_plot", action='store_true')

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)

    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args


if __name__ == "__main__":
    args = set_args()
    if args.filepath == ".pt":
        args.filepath = f'{args.dataset}-{args.model_name}-{args.model_weight}-freeze{args.freeze_layers}-batch{args.batch_size}-epoch{args.epochs}-lr{args.lr}.pt'
    set_seed(args.seed)
    #train(args)
    test(args)
    


