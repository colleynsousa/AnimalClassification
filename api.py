#+.+
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,random_split
from PIL import Image 
from fastapi import FastAPI,UploadFile,File
from fastapi.responses import HTMLResponse

class ResBlock(nn.Module): 
    def __init__(self,in_ch,out_ch): 
        super().__init__()
        
        self.conv=nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1,bias=False), #second convolution now in each block to refine features before adding skip connection
            nn.BatchNorm2d(out_ch)
        )
        
        self.skip=nn.Identity() #skip path returns input unchanged
        if in_ch!=out_ch: #if number of channels change
            self.skip=nn.Conv2d(in_ch,out_ch,1,bias=False) #use 1x1 convolution to change channel count to keep spatial size the same
        self.relu=nn.ReLU(inplace=True) #activation applied after combining learned features with input
        self.pool=nn.MaxPool2d(2) #downsample spatial size by 2 after residual addition
        
    def forward(self,x): 
        out=self.conv(x) #run input through main convolutional path
        skip=self.skip(x) #run input through skip connection 
        out=out+skip #residual additionL out=out+skip
        out=self.relu(out) #apply non-linearity after combining
        out=self.pool(out) #downsample
        return out

class ResCNN(nn.Module):
    def __init__(self,num_classes=90):
        super().__init__()
            
        self.features=nn.Sequential(
            ResBlock(3,64),
            ResBlock(64,128),
            ResBlock(128,256),
            ResBlock(256,512)
        )

        self.classifier=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512,num_classes)
        )

    def forward(self,x):
        x=self.features(x)
        return self.classifier(x)
 
device="cpu"    
model=ResCNN(num_classes=90)
state_dict=torch.load("ResCNN.pth",map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

with open("name of the animals.txt","r") as f: 
    class_names=[line.strip() for line in f]

transform=transforms.Compose([
    transforms.Resize((280,196)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5391,0.5254,0.4752],std=[0.2933,0.2845,0.3052])
])

app=FastAPI()
@app.post("/predict_animal")
async def predict(file: UploadFile=File(...)): 
    image=Image.open(file.file).convert("RGB")
    x=transform(image).unsqueeze(0)
    
    with torch.no_grad(): 
        logits=model(x)
        pred=logits.argmax(dim=1).item()
    
    pred_name=class_names[pred]
        
    return{"predicted_class": pred_name}

@app.get("/",response_class=HTMLResponse)
async def home():
    return """
    <html>
        <body>
            <h2>Upload an image to predict the animal</h2>
            <form action="/predict_animal" method="post" enctype="multipart/form-data">
                <input type="file" name="file">
                <button type="submit">Predict</button>
            </form>
        </body>
    </html>
    """
