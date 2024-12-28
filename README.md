# Employee Attrition Prediction - Machine Learning Web Application

## Overview

This project aims to create a machine learning model that predicts employee attrition (whether an employee will leave the company) based on various features. The model is integrated into a user-friendly web application using Flask, allowing users to input employee information and receive predictions instantly.

## Steps and Components

1.**Data Preparation and Model Building:**

**Data Collection:** Collected employee data from company records.
**Data Preprocessing:** Cleaned the data by handling missing values and transforming categorical variables.
**Feature Selection:** Selected relevant features and encoded categorical variables.
**Train-Test Split:** Divided the data into training and testing sets.
**Handling Imbalanced Data:** Used SMOTE to balance the target classes.
**Scaling:** Scaled numerical features using StandardScaler.
**Model Training:** Trained a Support Vector Machine (SVM) classifier on the training data.
**Model Evaluation:** Assessed model performance using accuracy and confusion matrix.


2. **Web Application Development:**
   - **Flask Setup:** Created a Flask app with routes for user input and predictions.
   - **HTML Templates:** Designed simple HTML forms for user interaction.
   - **CSS Styling:** Styled the web pages using CSS for an appealing and user-friendly interface.
   - **Form Handling:** Implemented routes to process user input and display predictions.
   - **File Upload Feature:** Added functionality to upload CSV files for batch predictions.

3. **Model Integration and Deployment:**
   - **Load and Train Model:** Implemented a function to load the trained SVM model.
   - **Process Input Data:** Handled user input, preprocessed it, and made predictions.
   - **Display Predictions:** Showed predictions on the webpage for immediate feedback.
   
4. **Local Testing and Deployment:**
   - **Local Testing:** Tested the application locally to ensure all features work correctly.
   - **Deployment:** Deployed the Flask web application on a cloud platform (e.g., Heroku).
   - **Sharing with Team:** Provided the deployed application URL for team testing and feedback.

## Future Enhancements

- **Model Improvement:** Explore different algorithms and hyperparameter tuning for better performance.
- **UI Enhancement:** Improve the user interface with additional features and visualizations.
- **Predictive Insights:** Offer insights or recommendations based on predictions.
- **User Authentication:**  Implement user authentication for secure access.

This project combines machine learning and web development to provide a practical solution for predicting employee attrition, helping HR departments make informed decisions. The web application simplifies the prediction process, making it a valuable tool for management.
