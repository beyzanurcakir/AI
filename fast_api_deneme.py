#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:17:41 2025

@author: beyzanurcakir
"""

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse


class ERP_Dataset(Dataset):
    def __init__(self, instructions, intents, responses, tokenizer, max_len=128):
        self.instructions = instructions
        self.intents = intents
        self.responses = responses
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.instructions[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.intents[idx], dtype=torch.long),
            "responses": self.responses[idx]  
        }


data = pd.read_csv('customer_training_data.csv')  


data['intent'] = data['intent'].astype('category').cat.codes  


MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

dataset = ERP_Dataset(data["instruction"].tolist(), data["intent"].tolist(), data["response"].tolist(), tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_dir="./logs",
    logging_steps=10
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)
"""trainer.train()"""



def create_pdf(response: str) -> BytesIO:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, "Cevap:")
    c.drawString(100, 730, response)
    c.showPage()
    c.save()
    buffer.seek(0)
    print("PDF dosyası oluşturuldu!")  
    return buffer


app = FastAPI()

class RequestData(BaseModel):
    query: str
    
@app.post("/classify_and_respond/")
def classify_and_respond(request: RequestData):
    
    import random
    predicted_intent = random.randint(0, 2)  

   
    intent_to_response_map = {
        0: "Siparişinizin durumu: Beklemede",
        1: "Evet, yeni sipariş verebilirsiniz.",
        2: "Ürün 2 gün içinde teslim edilecek."
    }
    response = intent_to_response_map.get(predicted_intent, "Yanıt bulunamadı.")

   
    pdf = create_pdf(response)


    return StreamingResponse(pdf, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=response.pdf"})



class RequestData(BaseModel):
    reportText: str = None  

@app.get("/calculate_length/")
async def calculate_length(reportText: str = None):
    if not reportText:  
        raise HTTPException(status_code=400, detail="Text giriniz")  
    text_length = len(reportText)  
    return JSONResponse(content={"text_length": text_length})  
"""
@app.post("/classify_and_respond/")
def classify_and_respond(request: RequestData):
    
    inputs = tokenizer(request.query, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predicted_intent = torch.argmax(outputs.logits, dim=1).item()
    
   
    intent_to_response_map = {
        0: "Siparişinizin durumu: Beklemede",
        1: "Evet, yeni sipariş verebilirsiniz.",
        2: "Ürün 2 gün içinde teslim edilecek."
    }
    response = intent_to_response_map.get(predicted_intent, "Yanıt bulunamadı.")
    
    
    pdf = create_pdf(response)
    
   
    return StreamingResponse(pdf, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=response.pdf"})"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
