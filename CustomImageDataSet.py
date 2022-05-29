
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

# write this filename as CustomImageDataSet
class FacesAndEmotions(torch.utils.data.Dataset):

    def __init__(self, transform=None, images=None, expressions=None):
        self.transform = transform
        self.images = images
        self.expressions = expressions

    def __len__(self):
        return len(self.images)

    def __getitem__(self, inx):
        img = self.images[inx]
        expression = self.expressions[inx]

        if self.transform:
            img = self.transform(img)
        
        #returning a tuple of images and its emotion 
        return (img, expression)





