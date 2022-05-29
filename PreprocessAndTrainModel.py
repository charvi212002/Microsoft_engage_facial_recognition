
#importing all the necessary libraries
import torch
import FaceEmotionDetectionModel 
from CustomImageDataSet import FacesAndEmotions
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

if not torch.cuda.is_available():
    from torchsummary import summary

# checking the availablity of GPU 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
shape = (44, 44)


class DataSet_preprocessing:

    def __init__(self):

        # FER2013 emotion dataset 
        # columns present ['emotion','pixels','Usage']
        # Usage -> ['Training','PublicTest','PrivateTest']

        fer1 = pd.read_csv("dataset/fer1.csv") 
        fer2 = pd.read_csv("dataset/fer2.csv") 
        fer3 = pd.read_csv("dataset/fer3.csv") 
        fer4 = pd.read_csv("dataset/fer4.csv") 
        fer5 = pd.read_csv("dataset/fer5.csv") 
        fer6 = pd.read_csv("dataset/fer6.csv") 
        fer7 = pd.read_csv("dataset/fer7.csv") 
        fer8 = pd.read_csv("dataset/fer8.csv") 
        data = pd.concat([fer1,fer2,fer3,fer4,fer5,fer6,fer7,fer8])
        data = data.reset_index(drop=True)
        data = data[['emotion','pixels','Usage']]
        #data = pd.read_csv("dataset/fer2013.csv")

        # storing the dataset into a pandas dataframe and preprocessing it
        # 1. converting the a string into a list of number
        # 2. converting that list into a 2D-array storing the pixels of an image 
        # 3. Dimension of the 2D-array is 48*48
        data['face'] = data['pixels'].apply(lambda x: [int(pixel) for pixel in x.split()])
        data['face'] = data['face'].apply(lambda x: np.asarray(x).reshape(48, 48))
        data['face'] = data['face'].apply(lambda x: x.astype('uint8'))
        data['face'] = data['face'].apply(lambda x: Image.fromarray(x))

        training = data[data['Usage']=='Training'].reset_index(drop=True)  
        public = data[data['Usage']=='PublicTest'].reset_index(drop=True)
        test = data[data['Usage']=='PrivateTest'].reset_index(drop=True)

        #combining two dataframes 'training' and 'public' to form a new dataframe: training
        training = pd.concat([training, public])

        images = list(training['face'])
        test_images = list(test['face'])
        expressions = list(training['emotion'])
        test_expressions =list(test['emotion'])

        print('training data size = {} , validation data size = {}'.format(
            len(images), len(test_images)))

        
        # Data Augmentation 
        train_transform = transforms.Compose([
            transforms.RandomCrop(shape[0]),
            transforms.RandomHorizontalFlip(),
            ToTensor(),
        ])

        validation_transform = transforms.Compose([
            transforms.CenterCrop(shape[0]),
            ToTensor(),
        ])

        self.training = FacesAndEmotions(transform=train_transform, 
                                images=images, 
                                expressions=expressions)

        self.test = FacesAndEmotions(transform=validation_transform, 
                              images=test_images, 
                              expressions=test_expressions)
        
def main():

    # Initializing the variables 
    batch_size = 128
    lr = 0.015                      # Learning Rate
    epochs = 300                    # Epochs
    lr_decay_start = 80             # Epoch at which Learning Rate starts to decay
    lr_decay_rate = 0.85            # Rate at which Learning Rate slowly decays
    lr_decay_every = 5              # Number of Epochs after which learning rate decays
    min_validation_loss = 10000

    # The classes of seven different emotions
    classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    network = FaceEmotionDetectionModel.Model(num_classes=len(classes)).to(device)           

    if not torch.cuda.is_available():
        summary(network, (1, shape[0], shape[1]))

    optimizer = torch.optim.SGD(network.parameters(), 
                                lr=lr, momentum=0.9, 
                                weight_decay=5e-3)
    
    criterion = nn.CrossEntropyLoss()
    dataset = DataSet_preprocessing()

    training_loader = DataLoader(dataset.training, 
                                 batch_size=batch_size, 
                                 shuffle=True, num_workers=1)
    
    validation_loader = DataLoader(dataset.test, 
                                  batch_size=batch_size, 
                                  shuffle=True, num_workers=1)

    for epoch in range(epochs):
        network.train()
        total = 0
        total_training_loss = 0
        correct = 0
        if epoch > lr_decay_start and lr_decay_start >= 0:
            current_lr = lr * (lr_decay_rate ** ((epoch - lr_decay_start) // lr_decay_every))
            for group in optimizer.param_groups:
                group['lr'] = current_lr
        else:
            current_lr = lr

        print('Learning Rate: %s' % str(current_lr))

        # printloss(training_loader ,'Training', 0, 0, 0,10000,epoch)
        for i, (x_train, y_train) in enumerate(training_loader):
            optimizer.zero_grad()

            x_train = x_train.to(device)
            y_train = y_train.to(device)

            y_predicted = network(x_train)

            loss = criterion(y_predicted, y_train)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(y_predicted.data, 1)

            total_training_loss += loss.data
            total += y_train.size(0)
            correct += predicted.eq(y_train.data).sum()

        accuracy = 100. * float(correct) / total
        print('Epoch [%d/%d] Training Loss: %.4f, Training Accuracy: %.4f' % (
            epoch + 1, epochs, total_training_loss / (i + 1), accuracy))

        network.eval()
        with torch.no_grad():
            # printloss(validation_loader ,'Validation', 0, 0, 0,10000,epoch)
            total = 0
            total_validation_loss = 0
            correct = 0

            for j, (x_val, y_val) in enumerate(validation_loader):

                x_val = x_val.to(device)
                y_val = y_val.to(device)

                #print(type(x_val))
                #print(x_val.shape)
                y_val_predicted = network(x_val)

                val_loss = criterion(y_val_predicted, y_val)
                _, predicted = torch.max(y_val_predicted.data, 1)

                total_validation_loss += val_loss.data
                total += y_val.size(0)
                correct += predicted.eq(y_val.data).sum()

            accuracy = 100. * float(correct) / total
            if total_validation_loss <= min_validation_loss:
                if epoch >= 200:
                    print('Save the Model')
                    model_states = {'net': network.model_states_dict()}
                    #print(len(model_states))
                    torch.save(model_states, 'model/test_model.t7' % (epoch + 1, accuracy))
                min_validation_loss = total_validation_loss

            print('Epoch [%d/%d] Validation Loss: %.4f, Validation Accuracy: %.4f' % (
                epoch + 1, epochs, total_validation_loss / (j + 1), accuracy))


if __name__ == "__main__":
  main()