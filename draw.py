from utils import *

from tkinter.filedialog import askopenfilename, asksaveasfilename
import pygame


@torch.no_grad()
def alter(image: np.array, model: EMA, t):
    cfg = model.config
    size = image.shape
    transforms = v2.Compose([
            v2.Resize(size=(cfg.imageSize, cfg.imageSize), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    image = transforms(torch.tensor(image / 255, dtype=torch.float32, device=DEVICE)).unsqueeze(0)
    image = image.transpose(3, 2)

    step = cfg.numTimesteps // cfg.evalTimesteps
    selected = list(range(cfg.numTimesteps - 1, -1, -step))[-t:]

    noise = torch.randn(*image.shape, device=DEVICE)
    _, aHat, _ = schedule(selected[0] + 1, cfg.numTimesteps)

    noised = math.sqrt(aHat) * image + math.sqrt(1 - aHat) * noise

    for i, t in enumerate(selected):
        _, aHat, _ = schedule(t + 1, cfg.numTimesteps)
        outputs = model(noised, torch.tensor(t, dtype=torch.long, device=DEVICE).repeat(1))

        x0 = (noised - math.sqrt(1 - aHat) * outputs) / math.sqrt(aHat)
        x0 = x0.clamp(-3, 3)

        if i == len(selected) - 1:
            noised = x0
            break

        tPrev = selected[i + 1]
        _, aHatPrev, _ = schedule(tPrev + 1, cfg.numTimesteps)

        alphaGap = aHat / aHatPrev
        betaGap = 1 - alphaGap

        mean = (math.sqrt(aHatPrev) * betaGap / (1 - aHat)) * x0 + (math.sqrt(alphaGap) * (1 - aHatPrev) / (1 - aHat)) * noised
        variance = betaGap * (1 - aHatPrev) / (1 - aHat)

        noised = mean + math.sqrt(variance) * torch.randn_like(noised)

    noised = noised.transpose(3, 2)
    mean = torch.tensor(IMAGENET_MEAN, device=noised.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=noised.device).view(-1, 1, 1)
    shifted = noised * std + mean
    shifted = shifted.clamp(0, 1).squeeze() * 255

    data = shifted.cpu().numpy().astype(np.uint8)
    img = Image.fromarray(data.transpose(1, 2, 0))

    resized = img.resize((size[2], size[1]), resample=Image.Resampling.LANCZOS)
    resized = np.array(resized)
    return resized.transpose(2, 0, 1)


DEVICE = "cuda"
CONFIG = Config().load(os.path.join("configs", "config.json"))
encoder = ViTEncoder(CONFIG)
ema = EMA(encoder, decay=CONFIG.emaDecay)

if os.path.exists("checkpoints") and len(glob(os.path.join("checkpoints", "*.pt"))) > 0:
    mostRecentRun = os.path.join("checkpoints", "recent.pt")
    print(f"\nLoading {mostRecentRun}...\n")
    runData = torch.load(mostRecentRun)

    ema.load_state_dict(runData["ema"])

ema.model.to(DEVICE)

imagePath = askopenfilename(filetypes=[["images", [".jpg", ".JPEG", ".png"]]])
if not imagePath:
    quit()

image = Image.open(imagePath)
imageData = np.array(image)
imageData = imageData[:, :, :3].transpose(2, 1, 0)
imageHistory = [imageData.copy()]
historySlice = 0

viewLocation = [0, 0]
viewZoom = 1

dragging = False
mousePosition = None

control = False
debug = False

brush = "circle"
brushSize = 64

lastBrush = None
brushStrength = 10

pygame.init()
window = pygame.display.set_mode((1200, 600))
draw = pygame.Surface((1200, 600))
clock = pygame.time.Clock()

uiFont = pygame.font.SysFont("Calibri", 16, bold=True)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LCTRL:
                control = True

            if event.key == pygame.K_LALT:
                debug = not debug

            if event.key == pygame.K_LEFTBRACKET:
                brushSize *= 2 / 3

            if event.key == pygame.K_RIGHTBRACKET:
                brushSize *= 3 / 2

            if event.key == pygame.K_y:
                historySlice = min(len(imageHistory) - 1, historySlice + 1)

            if event.key == pygame.K_z:
                historySlice = max(0, historySlice - 1)

            if event.key == pygame.K_MINUS:
                brushStrength -= 5

            if event.key == pygame.K_EQUALS:
                brushStrength += 5

            if event.key == pygame.K_r:
                if lastBrush is not None:
                    # TODO: Redo last draw
                    pass

            if event.key == pygame.K_b:
                brush = "square" if brush == "circle" else "circle"

            if event.key == pygame.K_s and control:
                fileName = asksaveasfilename(initialdir = "./",title = "Select file",filetypes = (("png files", "*.png"), ("jpeg files","*.jpg"), ("all files","*.*")))
                if fileName:
                    currentImage = imageHistory[historySlice]
                    img = Image.fromarray(currentImage.transpose(2, 1, 0))
                    img.save(fileName)

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LCTRL:
                control = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                currentImage = imageHistory[historySlice]

                topLeft = (viewLocation[0] - (currentImage.shape[1] * viewZoom) // 2 + 600, viewLocation[1] - (currentImage.shape[2] * viewZoom) // 2 + 300)
                brushCenter = mousePosition[0] - topLeft[0], mousePosition[1] - topLeft[1]
                brushCenter = brushCenter[0] / viewZoom, brushCenter[1] / viewZoom
                sliceX = int(brushCenter[0] - brushSize / 2)
                sliceY = int(brushCenter[1] - brushSize / 2)

                startX, startY = max(sliceX, 0), max(sliceY, 0)
                endX, endY = sliceX + int(brushSize), sliceY + int(brushSize)

                if endX > startX and endY > startY:
                    imageSlice = currentImage[:, startX: endX, startY: endY]
                    newSlice = alter(imageSlice, ema.model, brushStrength)
                    newImage = currentImage.copy()
                    newImage[:, startX: startX + newSlice.shape[1], startY: startY + newSlice.shape[2]] = newSlice

                    lastBrush = [sliceX, sliceY, brushSize, brushStrength]

                    imageHistory = imageHistory[:historySlice + 1]
                    imageHistory.append(newImage)
                    historySlice += 1

            if event.button == 2:
                dragging = True

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 2:
                dragging = False

        if event.type == pygame.MOUSEWHEEL:
            viewZoom *= 1.5 ** (event.y / abs(event.y))


    if dragging:
        newPos = pygame.mouse.get_pos()
        viewLocation[0] = viewLocation[0] + (newPos[0] - mousePosition[0])
        viewLocation[1] = viewLocation[1] + (newPos[1] - mousePosition[1])
    
    mousePosition = pygame.mouse.get_pos()

    window.fill((0, 0, 0))
    draw.fill((0, 0, 0))

    imageSurface = pygame.surfarray.make_surface(imageHistory[historySlice].transpose(1, 2, 0))
    imageSurface = pygame.transform.scale_by(imageSurface, viewZoom)
    draw.blit(imageSurface, (viewLocation[0] - imageSurface.get_rect().width // 2 + 600, viewLocation[1] - imageSurface.get_rect().height // 2 + 300))
    # TODO: Draw image

    if brush == "circle":
        pygame.draw.circle(draw, (255, 255, 255), mousePosition, radius=brushSize * viewZoom / 2, width=1)
    else:
        rect = pygame.Rect(mousePosition[0] - brushSize * viewZoom // 2, mousePosition[1] - brushSize * viewZoom // 2, 
                           brushSize * viewZoom, brushSize * viewZoom)
        pygame.draw.rect(draw, (255, 255, 255), rect=rect, width=1)

    height = 0
    locationText = uiFont.render(f"({viewLocation[0]:.2f}, {viewLocation[1]:.2f}) | {viewZoom:.2f}x", True, (255, 255, 255))
    draw.blit(locationText, (0, height))
    height += locationText.get_rect().height + 8
    brushText = uiFont.render(f"{brushStrength}/{CONFIG.evalTimesteps} | {brushSize}", True, (255, 255, 255))
    draw.blit(brushText, (0, height))
    height += brushText.get_rect().height + 8

    if debug:
        pass

    window.blit(draw, (0, 0))

    pygame.display.flip()
    clock.tick()

