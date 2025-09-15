# Smart India Hackathon 2025 Project Documentation

## Project Title:
**AI-Driven Multilingual Public Health Chatbot for Disease Awareness**

## Team:
Ayushman Trivedi & Team

## Repository:
[ayushmantrivedi/AI-driven-public-healthh-chatbot-for-diseases-awareness](https://github.com/ayushmantrivedi/AI-driven-public-healthh-chatbot-for-diseases-awareness)

---

## Problem Statement

India’s diversity in languages and communication channels (WhatsApp, SMS, mobile) limits timely access to healthcare information. Early diagnosis and disease awareness are hindered especially in rural and semi-urban regions due to language barriers and lack of medical expertise. Our project directly addresses this, offering an **AI-powered chatbot** accessible via WhatsApp and SMS, capable of accurate, language-agnostic disease prediction and health guidance.

---

## Solution Overview

We present a **deep learning pipeline** wrapped in a Flask API, powering an interactive chatbot accessible via WhatsApp, SMS, and the web. The chatbot takes user symptoms in multiple Indian languages, predicts likely diseases, and provides actionable medical guidance.

### Key Features
- **Multilingual Support:** Accepts user input in English, Hindi, Odia, Telugu, Tamil, and more. Dynamic stopword filtering and language detection.
- **Clinical Embeddings:** Uses Bio_ClinicalBERT for advanced symptom understanding.
- **Data Augmentation:** Synonym replacement for robust model generalization.
- **Feature Fusion:** Combines text/numeric features using neural attention mechanisms.
- **Global vs. Local Training:** Innovative line-search logic for improved generalization.
- **Reliable Evaluation:** K-Fold cross-validation for trustworthy metrics.
- **Production-Ready Artifacts:** Model weights, encoders, tokenizers, scalers, and label mappings saved for deployment.
- **Flask API Wrapper:** Allows easy integration as a RESTful service.
- **Chatbot Integration:** The API powers conversational interfaces for WhatsApp and SMS using Twilio or similar platforms.
- **Interactive Guidance:** Bot not only predicts disease but also offers general advice (e.g., when to visit a doctor, emergency tips) based on prediction confidence.

---

## Technical Architecture

### Data Preparation
- Reads health dataset (symptoms, disease labels).
- Cleans, normalizes, detects language, tokenizes input in supported languages.
- Augments training data with medical synonym replacement.
- Extracts/scales numeric health features.

### Embedding & Fusion
- Uses Bio_ClinicalBERT for symptom text encoding.
- Fuses textual and numeric data via attention-based neural network.

### Model Training
- Trains classifier using global-vs-local error line-search logic.
- Evaluates with K-Fold cross-validation for reliability.

### API & Chatbot Integration
- **Flask API:** Serves `/predict` endpoint accepting user symptoms and optional numeric data.
- **WhatsApp/SMS Integration:** Uses Twilio or similar gateway to connect the Flask API to messaging platforms.
- **Chatbot Flow:**
  - User sends symptoms (in any supported language) via WhatsApp/SMS.
  - Message is sent to Flask API.
  - API preprocesses input, runs model inference, returns disease prediction and confidence.
  - Bot replies with predicted disease, confidence score, and actionable health tips.
  - If confidence is low or symptoms are severe, bot advises contacting a healthcare provider immediately.

---

## SIH 2025 Innovation & Impact

### What makes this project innovative?
- **Multichannel, Multilingual Healthcare AI:** First-of-its-kind fusion of clinical NLP and health features for disease prediction on WhatsApp/SMS, accessible in major Indian languages.
- **Robust Generalization:** Line-search training logic prioritizes real-world validation.
- **Extensible & Explainable:** Modular, transparent, and easily adaptable to new diseases, languages, or channels.

### Societal Impact
- **Accessibility:** Delivers disease awareness and medical guidance to diverse linguistic and technological groups.
- **Scalability:** Reach millions via WhatsApp/SMS/mobile/web platforms.
- **Empowerment:** Enables timely self-assessment and health guidance, reducing disease burden and improving outcomes.

---

## Results & Evaluation

**K-Fold Cross-Validation Results:**
- **Average Loss:** 0.0359
- **Average Accuracy:** 99.00%
- **Average Macro F1 Score:** 0.9900

**Label Mapping:**
```
COVID-19    -> 0
Dengue      -> 1
Gastritis   -> 2
Influenza   -> 4
Malaria     -> 5
Measles     -> 6
Pneumonia   -> 7
TB          -> 8
Typhoid     -> 9
```

**Demo Prediction:**
- Input: 'fever and cough with headache'
- Output: **Malaria** (confidence: 0.641)

---

## How to Run

1. Prepare health dataset CSV (with `symptoms` and `disease` columns).
2. Configure paths and parameters in `sihdemo.py` as needed.
3. Run the script to train and save model artifacts:
   ```bash
   python sihdemo.py --csv /path/to/dataset.csv
   ```
4. **Flask API Setup**:  
   - Wrap the model in a Flask API, serving `/predict` endpoint.
   - Sample request:
     ```http
     POST /predict
     {
       "symptoms": "fever and cough with headache",
       "language": "en",
       "numeric_data": [optional]
     }
     ```
   - Response:
     ```json
     {
       "predicted_disease": "Malaria",
       "confidence": 0.641,
       "advice": "Please consult your nearest healthcare provider if symptoms persist or worsen."
     }
     ```
5. **WhatsApp/SMS Integration**:
   - Use Twilio or similar to route WhatsApp/SMS messages to Flask API.
   - Bot responds interactively with prediction and advice.

---

## Future Scope

- Expand to more languages (Punjabi, Bengali, Gujarati, etc.).
- Integrate video/image-based symptom checking in chatbot.
- Add explainability (symptom highlighting, confidence rationale).
- Deploy as voice-enabled chatbot for illiterate users.
- Integrate with government health platforms for mass outreach.

---

## Conclusion

Our AI-driven multilingual health chatbot, accessible via WhatsApp, SMS, and web, combines advanced NLP, robust training, and interactive guidance to deliver accessible, reliable disease awareness for India’s diverse population. Its scalability and societal relevance position it as a game-changer in public health for SIH 2025.

---

*For more details, contact ayushmantrivedi or refer to the [GitHub repository](https://github.com/ayushmantrivedi/AI-driven-public-healthh-chatbot-for-diseases-awareness).*