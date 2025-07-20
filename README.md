Generative AI Airline Passenger Satisfaction Dashboard
Overview
The Generative AI Airline Passenger Satisfaction Dashboard is an interactive web application that leverages Generative AI to deliver actionable insights from passenger satisfaction data. Built with Streamlit and hosted on Google Colab with ngrok for public access, the app uses distilgpt2 (Hugging Face Transformers) to generate natural language summaries and synthetic customer comments. Combined with a Random Forest model for satisfaction prediction, it offers a predictive chatbot and interactive visualizations, processing a 24-column dataset (train.csv) with passenger demographics, flight details, and satisfaction ratings.
Features

AI-Generated Summaries: Uses distilgpt2 to produce executive-style summaries of satisfaction trends (e.g., by class or age group), transforming raw statistics into natural language insights.
Synthetic Customer Comments: Generates realistic passenger feedback (e.g., “Wi-Fi was poor at 2/5, but seats were comfortable”) using distilgpt2, based on ratings and flight details.
Predictive Chatbot: Allows users to input passenger details (e.g., age, class, Wi-Fi rating) and predicts satisfaction with AI-generated explanations, powered by a Random Forest model and distilgpt2.
Interactive Visualizations: Includes Plotly-based bar charts (satisfaction by class/customer type), scatter plots (flight distance vs. age), and line charts (service ratings by class).
Data Preprocessing: Handles missing values, encodes categorical variables (OneHotEncoder), and scales numerical features (StandardScaler) using pandas.

Screenshots

Technical Details

Dataset: train.csv with 24 columns (e.g., id, Gender, Age, Class, satisfaction). A sample dataset is included for testing (1000 records).
Generative AI: distilgpt2 (Hugging Face Transformers) for text generation, optimized for Colab’s CPU constraints, producing natural language summaries and comments.
Machine Learning: Random Forest Classifier (scikit-learn) trained on 20 features (numerical: Age, Flight Distance; categorical: Gender, Class), achieving ~85–90% accuracy.
Technologies:
Python 3.11, Streamlit 1.29.0, pandas 2.2.2, scikit-learn 1.5.2, Plotly 5.24.1
Hugging Face Transformers 4.44.2, PyTorch 2.6.0
ngrok 7.1.6 for public URL deployment


Deployment: Hosted on Google Colab with ngrok for temporary public access. Suitable for Streamlit Community Cloud for permanent hosting.

Setup Instructions

Clone the Repository:git clone https://github.com/yourusername/airline-satisfaction-genai.git
cd airline-satisfaction-genai


Environment Setup:
Use Google Colab or a local Python 3.11 environment.
Install dependencies:pip install -r requirements.txt


Download ngrok (for Colab):wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar -xvzf ngrok-v3-stable-linux-amd64.tgz




ngrok Configuration:
Get a free authtoken from https://ngrok.com.
Run:./ngrok authtoken YOUR_NGROK_AUTHTOKEN




Generate or Use Dataset:
Use the included train.csv or generate a new one:python generate_dataset.py




Run the App:
In Colab or locally:streamlit run app.py


For Colab, expose with ngrok:from pyngrok import ngrok
ngrok.kill()
public_url = ngrok.connect(8501, bind_tls=True)
print("Your Streamlit app is live at:", public_url)




Test the App:
Access the app via the ngrok URL or localhost:8501 (local).
Upload train.csv, apply filters, and test Gen AI features (summaries, comments, chatbot) and visualizations.



Challenges and Solutions

Challenge: distilbert-base-uncased was unsupported for text generation.
Solution: Adopted distilgpt2, a lightweight causal language model compatible with transformers.pipeline("text-generation").


Challenge: Streamlit session state errors in Colab.
Solution: Ensured streamlit run app.py with ngrok tunneling for proper session state initialization.


Challenge: Chatbot input data mismatches causing ColumnTransformer errors.
Solution: Aligned input DataFrame with 20 feature columns used in training.


Challenge: Limited text generation quality with distilgpt2.
Solution: Tuned prompts and max_length parameters for better outputs within Colab’s constraints.



Gen AI Highlights

Summary Generation: distilgpt2 transforms statistical summaries into executive-style reports (e.g., “Business class passengers exhibit a 60% satisfaction rate, with Wi-Fi rated 2.8/5”).
Synthetic Comments: Converts structured data into realistic passenger feedback, enhancing interpretability.
Chatbot Explanations: Combines Random Forest predictions with distilgpt2-generated narratives to explain satisfaction factors (e.g., “Low Wi-Fi rating and delays led to dissatisfaction”).

Future Improvements

Integrate advanced LLMs (e.g., Grok 3 via xAI API) for enhanced text generation.
Deploy to Streamlit Community Cloud for permanent hosting.
Optimize distilgpt2 prompts for improved coherence.
Add real-time data streams for dynamic Gen AI insights.

Contact
For a live demo or inquiries, contact [Your Name] at [Your Email].Live demo available upon request (ngrok-based) or deployable via Streamlit Community Cloud.
