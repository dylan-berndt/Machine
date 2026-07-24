from .process import *

import torch
import torchvision.transforms.v2 as v2
import torchvision
from torch.utils.data import Dataset, DataLoader

import math

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def schedule(t, maxT, s=0.008):
    f = lambda x: math.cos((x / maxT + s) / (1 + s) * math.pi / 2) ** 2
    aHat = f(t) / f(0)
    aHatPrev = f(t - 1) / f(0) if t > 0 else 1.0
    a = aHat / aHatPrev
    b = min(1 - a, 0.999)
    a = 1 - b
    return a, aHat, b


class DiffusionData(Dataset):
    def __init__(self, config):
        self.config = config

        videoFrames = extractVideo(config.directory)
        images = extractImages(config.directory)
        print()

        self.images = videoFrames + images

        print(f"{len(self.images)} frames extracted: {len(videoFrames)} from video, {len(images)} from images")

        faceIdentifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        failed = []
        for i, image in enumerate(self.images):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = faceIdentifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                failed.append(i)
            else:
                x, y, w, h = faces[0]
                crop = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                self.images[i] = crop

            print(f"\rExamined image {i + 1} for faces", end="")

        print()

        lag = 0
        for i in failed:
            del self.images[i - lag]
            lag += 1

        print(f"{len(failed)} frames failed face extraction")

        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.RandomResizedCrop(size=(config.imageSize, config.imageSize), scale=(0.75, 1.0), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def __len__(self):
        return len(self.images) * self.config.numTimesteps

    def __getitem__(self, i):
        image = self.images[i // self.config.numTimesteps]
        t = i % self.config.numTimesteps

        image = self.transforms(image)

        noise = torch.randn(*image.shape)
        a, aHat, b = schedule((t + 1), self.config.numTimesteps)

        noised = math.sqrt(aHat) * image + math.sqrt(1 - aHat) * noise

        return image, noise, noised, torch.tensor(t, dtype=torch.long)


def generateImages(model, number=1):
    config = model.config
    image = torch.randn(number, 3, config.imageSize, config.imageSize)

    for i in range(0, config.numTimesteps, config.numTimesteps // config.evalTimesteps):
        print(image.min(), image.max(), i)
        t = config.numTimesteps - i - 1
        a, aHat, b = schedule(t + 1, config.numTimesteps)
        outputs = model(image, torch.tensor(t, dtype=torch.long).repeat(number))

        x0_pred = (image - math.sqrt(1 - aHat) * outputs) / math.sqrt(aHat)
        x0_pred = x0_pred.clamp(-3, 3)  # roughly your normalized data's valid range
        outputs = (image - math.sqrt(aHat) * x0_pred) / math.sqrt(1 - aHat)

        image = (1 / math.sqrt(a)) * (image - ((b / math.sqrt(1 - aHat)) * outputs))

        if i != config.numTimesteps - 1:
            noise = torch.randn(*image.shape)
            image += noise * math.sqrt(b)

    mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device).view(-1, 1, 1)
    return (image * std + mean).clamp(0, 1)
