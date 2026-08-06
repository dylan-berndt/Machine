from utils import *

import math
import wandb

DEVICE = "cuda"
CONFIG = Config().load(os.path.join("configs", "config.json"))

encoder = ViTEncoder(CONFIG)
encoder.to(DEVICE)
dataset = DiffusionData(CONFIG)

train = DataLoader(dataset, batch_size=CONFIG.batchSize, shuffle=True)

optimizer = torch.optim.Adam(encoder.parameters(), lr=CONFIG.learningRate)
ema = EMA(encoder, decay=CONFIG.emaDecay)

totalSteps = CONFIG.epochs * len(train)

def lrLambda(step):
    if step < CONFIG.warmupSteps:
        return step / max(1, CONFIG.warmupSteps)
    progress = (step - CONFIG.warmupSteps) / max(1, totalSteps - CONFIG.warmupSteps)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrLambda)

os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
os.environ["WANDB_START_METHOD"] = "thread"

step = 0
if os.path.exists("checkpoints") and len(glob(os.path.join("checkpoints", "*.pt"))) > 0:
    paths = glob(os.path.join("checkpoints", "*.pt"))
    mostRecentRun = sorted([(path, int(os.path.basename(path).removeprefix("step_").removesuffix(".pt"))) for path in paths], key=lambda x: x[1])[-1][0]
    print(f"\nLoading {mostRecentRun}...\n")
    runData = torch.load(mostRecentRun)

    run = wandb.init(entity="dylanberndt123-missouri-state-university", project="Machine", config=CONFIG.serialize(), id=runData["runID"], resume="must",
    settings=wandb.Settings(init_timeout=300))

    optimizer.load_state_dict(runData["optimizer"])
    scheduler.load_state_dict(runData["scheduler"])
    encoder.load_state_dict(runData["model"])
    ema.load_state_dict(runData["ema"])
    step = runData["step"]

else:
    os.makedirs("checkpoints", exist_ok=True)

    run = wandb.init(entity="dylanberndt123-missouri-state-university", project="Machine", config=CONFIG.serialize())

intervalLossSum = 0
intervalSteps = 0
for epoch in range(CONFIG.epochs):
    encoder.train()
    progress = 0
    for image, noise, noised, t in train:
        outputs = encoder(noised.to(DEVICE), t.to(DEVICE))
        loss = nn.functional.mse_loss(noise.to(DEVICE), outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update(encoder)

        progress += 1
        step += 1
        intervalLossSum += loss.item()
        intervalSteps += 1
        print(f"\rEpoch: {epoch + 1} | {progress}/{len(train)} training steps | Loss: {loss.item():.3f}", end="")

        if step % 1000 == 0:
            wandb.log({"Train Loss": intervalLossSum / intervalSteps}, step=step)
            intervalLossSum = 0
            intervalSteps = 0

            torch.save({
                "step": step,
                "epoch": epoch,
                "model": encoder.state_dict(),
                "ema": ema.state_dict(),
                "runID": run.id,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, os.path.join("checkpoints", f"step_{step}.pt"))

            with torch.no_grad():
                images = generateImages(ema.model, number=20)
                for i in range(images.shape[0]):
                    image = images[i]
                    torchvision.utils.save_image(image, os.path.join("results", f"image {i + 1}.png"))
                wandb.log({"Examples": wandb.Image(torchvision.utils.make_grid(images, 5, 0))}, step=step)

    print()
