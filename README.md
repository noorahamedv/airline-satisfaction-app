# ✈️ Generative AI Airline Passenger Satisfaction Dashboard

An interactive Streamlit application leveraging **Generative AI (distilgpt2)** and **Machine Learning (Random Forest)** to analyze and predict airline passenger satisfaction. Features AI-generated summaries, predictive chatbots, and interactive visualizations — hosted on **Google Colab + ngrok** for easy public access.

![Streamlit Screenshot](your-screenshot-link-here)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/noorahamedv/airline-satisfaction-genai/blob/main/app.ipynb)

---

## 🚀 Features

* **AI-Generated Summaries:** Executive-style insights using `distilgpt2` to explain satisfaction trends.
* **Synthetic Passenger Comments:** Converts data into realistic customer feedback.
* **Predictive Chatbot:** Predicts satisfaction with AI-generated explanations.
* **Interactive Visualizations:** Dynamic `Plotly` charts (bar, scatter, line plots) for data exploration.
* **End-to-End Data Pipeline:** Preprocessing, modeling, and visualization all-in-one Streamlit app.

---

## 🛠️ Tech Stack

| Component           | Technologies                                                         |
| ------------------- | -------------------------------------------------------------------- |
| **Language**        | Python 3.11                                                          |
| **Web App**         | Streamlit 1.29.0, ngrok                                              |
| **ML Models**       | Random Forest (scikit-learn), distilgpt2 (Hugging Face Transformers) |
| **Visualization**   | Plotly                                                               |
| **Data Processing** | pandas, OneHotEncoder, StandardScaler                                |
| **Deployment**      | Google Colab + ngrok (temporary) or Streamlit Community Cloud        |

---

## 📁 Dataset

* **train.csv**: 24 columns including demographics, flight details, and satisfaction ratings (\~1000 sample rows).
* Features: `Gender`, `Age`, `Class`, `Inflight Wi-Fi`, `Cleanliness`, `Satisfaction` and more.

---

## 💻 Setup Instructions

### 🔥 Quickstart

```bash
git clone https://github.com/noorahamedv/airline-satisfaction-genai.git
cd airline-satisfaction-genai
pip install -r requirements.txt
streamlit run app.py
```

### 🌐 Colab + ngrok Deployment

```bash
# Setup ngrok
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar -xvzf ngrok-v3-stable-linux-amd64.tgz
./ngrok authtoken YOUR_NGROK_AUTHTOKEN
streamlit run app.py
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(public_url)
```

---

## 📊 App Features Walkthrough

| Module                 | Description                                                                 |
| ---------------------- | --------------------------------------------------------------------------- |
| **Summaries**          | Executive-style summaries of satisfaction by class, age group.              |
| **Synthetic Comments** | Generates sample passenger comments from ratings.                           |
| **Predictive Chatbot** | Inputs custom passenger profile, predicts satisfaction, explains reasoning. |
| **Interactive Charts** | Plotly-based visuals for deeper exploration.                                |

---

## ⚙️ Challenges & Solutions

| Challenge                   | Solution                                         |
| --------------------------- | ------------------------------------------------ |
| `distilbert` unsupported    | Switched to `distilgpt2` for generation tasks    |
| Colab session errors        | Streamlit + ngrok for consistent session state   |
| Chatbot column misalignment | Data alignment to model feature schema           |
| Limited text quality        | Tuned prompts, sampling parameters for coherence |

---

## ✨ Future Roadmap

* [ ] Deploy on Streamlit Community Cloud 🌐
* [ ] Upgrade to advanced LLMs (e.g., GPT-4, xAI API)
* [ ] Integrate real-time data streams 📈
* [ ] Expand dataset and improve visual insights

---

## 📎 Live Demo

https://4a2d71a29327.ngrok-free.app/
