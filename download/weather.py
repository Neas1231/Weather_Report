#!/usr/bin/env python
# coding: utf-8
import scipy
import os
from PIL import Image
import streamlit as st
import pandas as pd
from st_on_hover_tabs import on_hover_tabs
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
#from pydrive.auth import GoogleAuth
#from pydrive.drive import GoogleDrive

#gauth = GoogleAuth()
#drive = GoogleDrive(gauth)

dictuar = {0:'Ясно' ,1:'Облачно' ,2:'Туман' ,3:'Мороз' ,4:'Град' ,5:'Молния',6: 'Нет погоды' , 7:'Дождь', 8:'Радуга',
           9:'Снег', 10:'Восход'}

st.set_page_config(layout="wide")
st.header("Программа для классификации погодных условий ")
st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)
device = torch.device("cpu")
model = models.efficientnet_b4()
num_ftrs = model.classifier[1].in_features
model.fc = nn.Linear(num_ftrs, 11)
model = model.to(device)
model.load_state_dict(torch.load('weights-no_weather_class-efinetb4-bestloss.pth',map_location ='cpu'))


uploaded_file = st.file_uploader('Загрузите картинку погоды, которую попробует распознать нейросеть!')
if uploaded_file is not None:

       bytes_data = uploaded_file.getvalue()
       
       image = Image.open(uploaded_file).convert('RGB')
       image_copy = image.copy()
       transform = transforms.Compose([
        transforms.Resize((128 , 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
       ])
       image = transform(image)
       
       model.eval()
       preds = model(image.unsqueeze(0))
       #st.write(preds.argmax())
       if st.button('Показать результат'):
          st.write(dictuar[int(preds.argmax())])
          #a = preds.max()
          #preds[preds == a] = 0
          #st.write(preds.argmax())
          #if preds[preds >=0.9].any():
              #st.write(dictuar[int(preds.argmax())])




       with st.sidebar:
               tabs = on_hover_tabs(tabName=['Предсказание', 'Картинка', 'Все вместе'], 
               iconName=['dashboard', 'money', 'economy'],
               styles = {'navtab': {'background-color':'#111',
                                           'color': '#818181',
                                           'font-size': '18px',
                                           'transition': '.3s',
                                           'white-space': 'nowrap',
                                           'text-transform': 'uppercase'},
                         'iconStyle':{'position':'fixed',
                                           'left':'7.5px',
                                           'text-align': 'left'},
                         'tabStyle' : {'list-style-type': 'none',
                                            'margin-bottom': '30px',
                                            'padding-left': '30px'}},
                                 key="1")
                         

       if tabs =='Предсказание':
           st.header(dictuar[int(preds.argmax())])
       if  tabs =='Картинка':
           st.image(image_copy)
       if tabs == 'Все вместе':
           st.header(dictuar[int(preds.argmax())]) 
           st.image(image_copy)
