# Automated Reporting System with BERT-based NLP

Havelsan destekli geliştirilen bitirme projesi. Müşteri taleplerini (instruction) 
BERT tabanlı bir NLP modeliyle sınıflandırıp, otomatik yanıt üreten ve bu yanıtı 
PDF rapor olarak sunan bir REST API servisi.

## Proje Amacı
ERP/müşteri hizmetleri senaryosunda gelen serbest metin taleplerini otomatik olarak 
sınıflandırıp (örn. sipariş durumu sorgusu, yeni sipariş talebi), uygun yanıtı 
otomatik oluşturmak ve raporlamak.

## Kullanılan Teknolojiler
- **BERT** (`bert-base-uncased`, HuggingFace Transformers) — intent classification
- **FastAPI** — REST API servisi
- **PyTorch** — model eğitimi ve inference
- **ReportLab** — otomatik PDF rapor oluşturma

## Mimari
1. Müşteri talebi API'ye gönderilir (`/classify_and_respond/`)
2. BERT modeli talebi 3 kategoriden birine sınıflandırır (fine-tuned `BertForSequenceClassification`)
3. Sınıflandırma sonucuna göre otomatik yanıt metni oluşturulur
4. Yanıt, PDF formatında rapor olarak kullanıcıya sunulur

## Durum / Notlar
- Model eğitim altyapısı (`Trainer`, `TrainingArguments`) kurulu; fine-tuning adımı 
  şu an devre dışı bırakılmış durumda (geliştirme/demo aşaması)
- API endpoint'i şu an demo amaçlı rastgele sınıflandırma kullanıyor; gerçek BERT 
  inference kodu hazır, aktivasyonu bekliyor

## Gelecek Planları
- RAG (Retrieval-Augmented Generation) tabanlı sisteme genişletme (Llama3/Mistral, 
  ChromaDB ile) — ulusal konferans sunumu hedefleniyor
