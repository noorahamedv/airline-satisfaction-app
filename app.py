```python
   import streamlit as st
   import pandas as pd
   import plotly.express as px
   import plotly.graph_objects as go
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.preprocessing import StandardScaler, OneHotEncoder
   from sklearn.compose import ColumnTransformer
   from sklearn.pipeline import Pipeline
   from sklearn.metrics import accuracy_score, classification_report
   from transformers import pipeline
   import torch
   import os

   # Streamlit app configuration
   st.set_page_config(page_title="Airline Passenger Satisfaction App", layout="wide")
   st.title("✈️ Airline Passenger Satisfaction Dashboard")
   st.markdown("Upload `train.csv` to explore insights, generate comments, and interact with a chatbot. [Download sample dataset](https://example.com/train.csv)")

   # Initialize DistilBERT (cached for performance)
   @st.cache_resource
   def load_nlp_model():
       return pipeline("text-generation", model="distilbert-base-uncased", device=-1)  # CPU

   nlp = load_nlp_model()

   # Sidebar for file upload and filters
   st.sidebar.header("Data Upload & Filters")
   uploaded_file = st.sidebar.file_uploader("Upload train.csv", type="csv")

   # Initialize session state
   if "data" not in st.session_state:
       st.session_state.data = None
   if "model" not in st.session_state:
       st.session_state.model = None

   # Load and preprocess data
   def load_and_preprocess_data(file):
       df = pd.read_csv(file)
       df = df.dropna()
       numeric_cols = ["Age", "Flight Distance", "Inflight wifi service", "Departure/Arrival time convenient",
                       "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
                       "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
                       "Baggage handling", "Checkin service", "Inflight service", "Cleanliness",
                       "Departure Delay in Minutes", "Arrival Delay in Minutes"]
       for col in numeric_cols:
           df[col] = pd.to_numeric(df[col], errors="coerce")
       df["Age Group"] = pd.cut(df["Age"], bins=[0, 20, 40, 60, 100], labels=["<20", "20-40", "40-60", "60+"])
       df["Distance Group"] = pd.cut(df["Flight Distance"], bins=[0, 500, 1500, 5000], labels=["<500km", "500-1500km", "1500+km"])
       return df

   # Train Random Forest model
   def train_model(df):
       X = df.drop(["id", "satisfaction", "Age Group", "Distance Group"], axis=1)
       y = df["satisfaction"].map({"satisfied": 1, "neutral or dissatisfied": 0})
       categorical_cols = ["Gender", "Customer Type", "Type of Travel", "Class"]
       numeric_cols = [col for col in X.columns if col not in categorical_cols]
       preprocessor = ColumnTransformer(
           transformers=[
               ("num", StandardScaler(), numeric_cols),
               ("cat", OneHotEncoder(), categorical_cols)
           ])
       model = Pipeline([
           ("preprocessor", preprocessor),
           ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
       ])
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       model.fit(X_train, y_train)
       y_pred = model.predict(X_test)
       st.write("**Model Performance**")
       st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
       st.write("Classification Report:")
       st.write(classification_report(y_test, y_pred, target_names=["Dissatisfied", "Satisfied"]))
       return model

   # Generate summary
   def generate_summary(df, filter_col, filter_val):
       filtered_df = df[df[filter_col] == filter_val]
       satisfaction_rate = len(filtered_df[filtered_df["satisfaction"] == "satisfied"]) / len(filtered_df) * 100
       avg_ratings = filtered_df[["Inflight wifi service", "Seat comfort", "Inflight entertainment"]].mean().to_dict()
       template = f"""
       {filter_col} {filter_val} passengers have a {satisfaction_rate:.1f}% satisfaction rate.
       Average ratings: Wi-Fi ({avg_ratings['Inflight wifi service']:.1f}/5), Seat comfort ({avg_ratings['Seat comfort']:.1f}/5),
       Inflight entertainment ({avg_ratings['Inflight entertainment']:.1f}/5). Key trends: {filter_val} passengers are {'more' if satisfaction_rate > 50 else 'less'} satisfied.
       """
       prompt = f"Refine this summary to sound natural and executive-style (100-150 words): {template}"
       try:
           result = nlp(prompt, max_length=200, num_return_sequences=1, truncation=True)[0]["generated_text"]
           return result
       except Exception as e:
           return f"Error generating summary: {str(e)}. Using template:\n{template}"

   # Generate synthetic comments
   def generate_synthetic_comment(row):
       prompt = f"""
       Convert these scores into a realistic customer comment (50-100 words):
       Inflight wifi: {row["Inflight wifi service"]}/5, Seat comfort: {row["Seat comfort"]}/5,
       Inflight entertainment: {row["Inflight entertainment"]}/5, Departure Delay: {row["Departure Delay in Minutes"]} minutes,
       Class: {row["Class"]}, Satisfaction: {row["satisfaction"]}.
       """
       try:
           result = nlp(prompt, max_length=150, num_return_sequences=1, truncation=True)[0]["generated_text"]
           return result
       except Exception as e:
           return f"Error generating comment: {str(e)}. Fallback: Wi-Fi was {row['Inflight wifi service']}/5, seats {row['Seat comfort']}/5."

   # Satisfaction explanation chatbot
   def explain_satisfaction(row, model):
       X = row.drop(["id", "satisfaction", "Age Group", "Distance Group"], errors="ignore")
       pred = model.predict(X)[0]
       pred_label = "satisfied" if pred == 1 else "neutral or dissatisfied"
       feature_importance = pd.DataFrame({
           "feature": model.named_steps["preprocessor"].get_feature_names_out(),
           "importance": model.named_steps["classifier"].feature_importances_
       }).sort_values("importance", ascending=False).head(3)
       template = f"""
       Passenger predicted as {pred_label}. Details: Age {row["Age"]}, Class {row["Class"]},
       Wi-Fi {row["Inflight wifi service"]}/5, Seat comfort {row["Seat comfort"]}/5,
       Delay {row["Departure Delay in Minutes"]} minutes. Top factors: {feature_importance.to_dict(orient="records")}.
       """
       prompt = f"Explain this prediction in a natural, concise way (100-150 words): {template}"
       try:
           result = nlp(prompt, max_length=200, num_return_sequences=1, truncation=True)[0]["generated_text"]
           return result
       except Exception as e:
           return f"Error generating explanation: {str(e)}. Fallback: Predicted {pred_label} due to {feature_importance['feature'].iloc[0]}."

   # Load data if uploaded
   if uploaded_file:
       st.session_state.data = load_and_preprocess_data(uploaded_file)
       st.session_state.model = train_model(st.session_state.data)

   # Filters
   if st.session_state.data is not None:
       df = st.session_state.data
       st.sidebar.subheader("Filters")
       class_filter = st.sidebar.selectbox("Class", ["All"] + list(df["Class"].unique()))
       customer_type_filter = st.sidebar.selectbox("Customer Type", ["All"] + list(df["Customer Type"].unique()))
       satisfaction_filter = st.sidebar.selectbox("Satisfaction", ["All"] + list(df["satisfaction"].unique()))

       # Apply filters
       filtered_df = df
       if class_filter != "All":
           filtered_df = filtered_df[filtered_df["Class"] == class_filter]
       if customer_type_filter != "All":
           filtered_df = filtered_df[filtered_df["Customer Type"] == customer_type_filter]
       if satisfaction_filter != "All":
           filtered_df = filtered_df[filtered_df["satisfaction"] == satisfaction_filter]

       # Tabs for features
       tab1, tab2, tab3, tab4 = st.tabs(["Summary Reports", "Synthetic Comments", "Chatbot", "Visualizations"])

       with tab1:
           st.header("Automated Summary Reports")
           filter_col = st.selectbox("Select filter for summary", ["Class", "Age Group", "Distance Group"])
           filter_val = st.selectbox("Select value", df[filter_col].unique())
           if st.button("Generate Summary"):
               summary = generate_summary(filtered_df, filter_col, filter_val)
               st.write(summary)

       with tab2:
           st.header("Synthetic Customer Comments")
           sample_rows = filtered_df.sample(3, random_state=42) if len(filtered_df) >= 3 else filtered_df
           for idx, row in sample_rows.iterrows():
               comment = generate_synthetic_comment(row)
               st.subheader(f"Passenger ID: {row['id']}")
               st.write(comment)

       with tab3:
           st.header("Satisfaction Explanation Chatbot")
           st.write("Enter passenger details to get an AI explanation of satisfaction prediction.")
           with st.form("chatbot_form"):
               age = st.number_input("Age", min_value=0, max_value=100, value=30)
               class_input = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
               wifi = st.slider("Inflight Wi-Fi Service (1-5)", 1, 5, 3)
               seat = st.slider("Seat Comfort (1-5)", 1, 5, 3)
               delay = st.number_input("Departure Delay (minutes)", min_value=0, value=0)
               submitted = st.form_submit_button("Explain Satisfaction")
               if submitted and st.session_state.model:
                   input_data = pd.DataFrame({
                       "Age": [age], "Class": [class_input], "Inflight wifi service": [wifi],
                       "Seat comfort": [seat], "Departure Delay in Minutes": [delay],
                       "Gender": ["Male"], "Customer Type": ["Loyal Customer"], "Type of Travel": ["Business travel"],
                       "Flight Distance": [1000], "Departure/Arrival time convenient": [3], "Ease of Online booking": [3],
                       "Gate location": [3], "Food and drink": [3], "Online boarding": [3], "Inflight entertainment": [3],
                       "On-board service": [3], "Leg room service": [3], "Baggage handling": [3], "Checkin service": [3],
                       "Inflight service": [3], "Cleanliness": [3], "Arrival Delay in Minutes": [0]
                   })
                   explanation = explain_satisfaction(input_data, st.session_state.model)
                   st.write(explanation)

       with tab4:
           st.header("Visualizations")
           # Bar chart
           bar_data = filtered_df.groupby(["Class", "Customer Type", "satisfaction"]).size().unstack(fill_value=0)
           bar_data = bar_data.div(bar_data.sum(axis=1), axis=0) * 100
           bar_data = bar_data.reset_index()
           fig_bar = px.bar(bar_data, x="Class", y=["satisfied", "neutral or dissatisfied"],
                            color_discrete_map={"satisfied": "#4CAF50", "neutral or dissatisfied": "#F44336"},
                            facet_col="Customer Type", title="Satisfaction by Class and Customer Type")
           st.plotly_chart(fig_bar, use_container_width=True)

           # Scatter plot
           fig_scatter = px.scatter(filtered_df, x="Age", y="Flight Distance", color="satisfaction",
                                    color_discrete_map={"satisfied": "#4CAF50", "neutral or dissatisfied": "#F44336"},
                                    title="Flight Distance vs. Age")
           st.plotly_chart(fig_scatter, use_container_width=True)

           # Line chart
           line_data = filtered_df.groupby("Class")[["Inflight wifi service", "Seat comfort", "Inflight entertainment"]].mean().reset_index()
           fig_line = go.Figure()
           for col in ["Inflight wifi service", "Seat comfort", "Inflight entertainment"]:
               fig_line.add_trace(go.Scatter(x=line_data["Class"], y=line_data[col], mode="lines+markers", name=col))
           fig_line.update_layout(title="Average Service Ratings by Class", xaxis_title="Class", yaxis_title="Average Rating")
           st.plotly_chart(fig_line, use_container_width=True)

   else:
       st.write("Please upload a CSV file to begin.")
   ```