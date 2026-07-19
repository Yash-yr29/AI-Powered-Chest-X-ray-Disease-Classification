import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

st.set_page_config(
    page_title="AI Powered Chest X-ray Disease Classification",
    page_icon="🩺",
    layout="centered"
)

st.markdown("""
<style>

.main{
    background:#f7f9fc;
}

h1{
    text-align:center;
    color:#0B5ED7;
}

.stButton>button{
    width:100%;
    border-radius:10px;
}

</style>
""",unsafe_allow_html=True)

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1=nn.Conv2d(3,16,3,1,1)
        self.relu1=nn.ReLU()
        self.pool1=nn.MaxPool2d(2,2)

        self.conv2=nn.Conv2d(16,32,3,1,1)
        self.relu2=nn.ReLU()
        self.pool2=nn.MaxPool2d(2,2)

        self.conv3=nn.Conv2d(32,64,3,1,1)
        self.relu3=nn.ReLU()
        self.pool3=nn.MaxPool2d(2,2)
        self.flatten=nn.Flatten()
        self.fc1=nn.Linear(64*28*28,128)
        self.relu4=nn.ReLU()
        self.dropout=nn.Dropout(0.5)
        self.fc2=nn.Linear(128,3)

    def forward(self,x):

        x=self.pool1(self.relu1(self.conv1(x)))
        x=self.pool2(self.relu2(self.conv2(x)))
        x=self.pool3(self.relu3(self.conv3(x)))
        x=self.flatten(x)
        x=self.relu4(self.fc1(x))
        x=self.dropout(x)
        x=self.fc2(x)
        return x


model=CNN()

from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "cnn_chest_xray.pth"

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu")
    )
)
model.eval()

transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])

classes=["Normal","Pneumonia","Tuberculosis"]

def predict(image):

    image=image.convert("RGB")
    image=transform(image)
    image=image.unsqueeze(0)

    with torch.no_grad():
        outputs=model(image)
        probabilities=torch.softmax(outputs,dim=1)
        confidence,predicted=torch.max(probabilities,1)
    return classes[predicted.item()],confidence.item()*100


st.sidebar.title("About")

st.sidebar.write("""
**Model**
Custom CNN
**Framework**
PyTorch
**Frontend**
Streamlit
""")

st.sidebar.markdown("---")
st.sidebar.write("**Developer**")
st.sidebar.write("Yash Raj")
st.sidebar.write("B.Tech CSE")
st.sidebar.write("IIIT Nagpur")


st.markdown("""
<h1 style="font-size:34px;">
🩺 AI Powered Chest X-ray Disease Classification
</h1>
""",unsafe_allow_html=True)

st.markdown(
"<p style='text-align:center;color:gray;font-size:15px;'>Deep Learning • PyTorch • CNN</p>",
unsafe_allow_html=True
)

uploaded_file=st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg","jpeg","png"],
    help="Supported formats: JPG, JPEG and PNG"
)

if uploaded_file:
    image=Image.open(uploaded_file)
    col1,col2=st.columns([1,1])

    with col1:

        st.image(
            image,
            caption="Uploaded X-ray",
            width=280
        )

    with st.spinner("Analyzing..."):
        prediction,confidence=predict(image)

    with col2:
        st.success("Prediction")

        st.markdown(
            f"<h2 style='color:#0B5ED7'>{prediction}</h2>",
            unsafe_allow_html=True
        )

        st.write("Confidence")
        st.progress(confidence/100)

        st.markdown(
            f"<h3>{confidence:.2f}%</h3>",
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption(
"⚠ This AI model is developed for educational and research purposes only. Predictions should not be considered a medical diagnosis. Please consult a qualified healthcare professional."
)
st.markdown(
"""
<center>
Developed with ❤️ by <b>Yash Raj</b><br>
B.Tech CSE • IIIT Nagpur
</center>
""",

unsafe_allow_html=True
)