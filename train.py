from utils import *

import math
# import wandb

DEVICE = "xpu"
CONFIG = Config().load(os.path.join("configs", "config.json"))

# wandb.init(entity="dylanberndt123-missouri-state-university", project="Machine", config=CONFIG.serialize())

encoder = ViTEncoder(CONFIG)
encoder.to(DEVICE)
dataset = DiffusionData(CONFIG)

train, test = torch.utils.data.random_split(dataset, [0.8, 0.2])
train = DataLoader(train, batch_size=CONFIG.batchSize, shuffle=True)
test = DataLoader(test, batch_size=CONFIG.batchSize, shuffle=True)

optimizer = torch.optim.Adam(encoder.parameters(), lr=CONFIG.learningRate)

totalSteps = CONFIG.epochs * len(train)

def lrLambda(step):
    if step < CONFIG.warmupSteps:
        return step / max(1, CONFIG.warmupSteps)
    progress = (step - CONFIG.warmupSteps) / max(1, totalSteps - CONFIG.warmupSteps)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrLambda)

step = 0
for epoch in range(CONFIG.epochs):
    encoder.train()
    progress = 0
    trainLossSum = 0
    for image, noise, noised, t in train:
        outputs = encoder(noised.to(DEVICE), t.to(DEVICE))
        loss = nn.functional.mse_loss(noise.to(DEVICE), outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        progress += 1
        trainLossSum += loss.item()
        print(f"\rEpoch: {epoch + 1} | {progress}/{len(train)} training steps | Loss: {loss.item():.3f}", end="")

    print()
    step += 1
    # wandb.log({"train/loss": trainLossSum / len(train), "epoch": epoch + 1}, step=step)

    with torch.no_grad():
        encoder.eval()
        progress = 0
        testLossSum = 0
        for image, noise, noised, t in test:
            outputs = encoder(noised.to(DEVICE), t.to(DEVICE))
            loss = nn.functional.mse_loss(noise.to(DEVICE), outputs)

            progress += 1
            testLossSum += loss.item()
            print(f"\rEpoch: {epoch + 1} | {progress}/{len(test)} testing steps | Loss: {loss.item():.3f}", end="")

        print()
        # wandb.log({"test/loss": testLossSum / len(test), "epoch": epoch + 1}, step=step)

        images = generateImages(encoder, number=20)
        for i in range(images.shape[0]):
            image = images[i]
            torchvision.utils.save_image(image, os.path.join("results", f"image {i + 1}.png"))

        # wandb.log({"samples": [wandb.Image(images[i]) for i in range(images.shape[0])]}, step=step)
