import asyncio
import io
import sys
import os
import torch
import torchvision.transforms as transforms
from projects.VolcanoFinder.models import MyFirstCNN

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask

from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "cnns"))

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MyFirstCNN().to(device)
model.load_state_dict(torch.load(
    os.path.join(os.path.dirname(__file__), "..", "notebooks", "8607BCE.pth"),
    map_location=device
))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None, "image_url": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Save uploaded image to static/ so frontend can show it
    image_path = os.path.join("static", file.filename)
    image.save(image_path)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        prediction = (output > 0.5).float().item()

    result = "🌋 Volcano" if prediction == 1.0 else "❌ Not a Volcano"

    async def cleanup_with_delay():
        await asyncio.sleep(4)
        os.remove(image_path)

    background_task = BackgroundTask(cleanup_with_delay)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "image_url": "/" + image_path
        },
        background=background_task
    )
# uvicorn app:app --reload || to launch the app