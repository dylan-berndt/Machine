from utils import *

DEVICE = "cpu"
CONFIG = Config().load(os.path.join("configs", "config.json"))

encoder = ViTEncoder(CONFIG)
encoder.to(DEVICE)
dataset = DiffusionData(CONFIG)

train, test = torch.utils.data.random_split(dataset, [0.8, 0.2])
train = DataLoader(train, batch_size=CONFIG.batchSize, shuffle=True)
test = DataLoader(test, batch_size=CONFIG.batchSize, shuffle=True)

optimizer = torch.optim.Adam(encoder.parameters(), lr=CONFIG.learningRate)

for epoch in range(CONFIG.epochs):
    with torch.no_grad():
        encoder.eval()
        images = generateImages(encoder, number=20)
        for i in range(images.shape[0]):
            image = images[i]
            torchvision.utils.save_image(image, os.path.join("results", f"image {i + 1}.png"))

    encoder.train()
    progress = 0
    for image, noise, noised, t in train:
        outputs = encoder(noised.to(DEVICE), t.to(DEVICE))
        loss = nn.functional.mse_loss(noise.to(DEVICE), outputs)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        optimizer.step()

        progress += 1
        print(f"\r{progress}/{len(train)} training steps | Loss: {loss.item():.3f}", end="")

    print()

    with torch.no_grad():
        encoder.eval()
        progress = 0
        for image, noise, noised, t in test:
            outputs = encoder(noised.to(DEVICE), t.to(DEVICE))
            loss = nn.functional.mse_loss(noise.to(DEVICE), outputs)

            progress += 1
            print(f"\r{progress}/{len(test)} testing steps | Loss: {loss.item():.3f}", end="")

        print()

        images = generateImages(encoder, number=20)
        for i in range(images.shape[0]):
            image = images[i]
            torchvision.utils.save_image(image, os.path.join("results", f"image {i + 1}.png"))
    