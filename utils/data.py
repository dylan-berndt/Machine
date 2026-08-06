from .process import *

import torch
import torchvision.transforms.v2 as v2
import torchvision
from torch.utils.data import Dataset, DataLoader

import math

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
FACE_THRESHOLD = 2.5


def schedule(t, maxT, s=0.008):
    f = lambda x: math.cos((x / maxT + s) / (1 + s) * math.pi / 2) ** 2
    aHat = f(t) / f(0)
    aHat = max(aHat, 1e-4)
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
        self.images = self.images

        print(f"{len(self.images)} frames extracted: {len(videoFrames)} from video, {len(images)} from images")

        faceIdentifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        failed = []
        for i, image in enumerate(self.images):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces, rejectLevels, levelWeights = faceIdentifier.detectMultiScale3(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                outputRejectLevels=True
            )

            if len(faces) == 0:
                failed.append(i)
            else:
                best_idx = int(levelWeights.argmax())  # highest-confidence detection
                x, y, w, h = faces[best_idx]
                confidence = levelWeights[best_idx]
                if confidence < FACE_THRESHOLD:  # tune empirically, e.g. try 2.0-4.0 as a starting point
                    failed.append(i)
                else:
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
            # v2.RandomPerspective(),
            v2.Resize(size=(config.imageSize, config.imageSize), antialias=True),
            v2.RandomResizedCrop(size=(config.imageSize, config.imageSize), scale=(0.75, 1.0), antialias=True),
            # v2.RandomRotation((-90, 90)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(0.1, 0.1, 0.1, 0.1),
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
    device = next(model.parameters()).device
    image = torch.randn(number, 3, config.imageSize, config.imageSize, device=device)

    step = config.numTimesteps // config.evalTimesteps
    selected = list(range(config.numTimesteps - 1, -1, -step))
    
    for i, t in enumerate(selected):
        _, aHat, _ = schedule(t + 1, config.numTimesteps)
        outputs = model(image, torch.tensor(t, dtype=torch.long, device=device).repeat(number))

        x0 = (image - math.sqrt(1 - aHat) * outputs) / math.sqrt(aHat)
        x0 = x0.clamp(-3, 3)

        if i == len(selected) - 1:
            image = x0
            break

        tPrev = selected[i + 1]
        _, aHatPrev, _ = schedule(tPrev + 1, config.numTimesteps)

        alphaGap = aHat / aHatPrev
        betaGap = 1 - alphaGap

        mean = (math.sqrt(aHatPrev) * betaGap / (1 - aHat)) * x0 + (math.sqrt(alphaGap) * (1 - aHatPrev) / (1 - aHat)) * image
        variance = betaGap * (1 - aHatPrev) / (1 - aHat)

        image = mean + math.sqrt(variance) * torch.randn_like(image)

    mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device).view(-1, 1, 1)
    return (image * std + mean).clamp(0, 1)
