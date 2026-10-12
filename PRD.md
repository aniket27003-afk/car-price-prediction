# Product Requirements Document (PRD)
## Car Price Prediction System Using Machine Learning

**Version:** 1.0  
**Project Type:** Machine Learning / Data Science Web Application  
**Status:** Planned  
**Target Audience:** Used-car buyers, sellers, and automotive enthusiasts

---

## 1. Product Overview

### 1.1 Product Name
**AutoValue AI — Smart Car Price Predictor**

### 1.2 Product Description
AutoValue AI is a machine learning-powered web application that estimates the selling price of a used car based on its specifications, condition-related attributes, and usage history.

The system will analyze historical car sales data, preprocess the dataset, train and evaluate regression models, and provide estimated prices through a simple, interactive web interface.

### 1.3 Problem Statement
Determining the fair selling price of a used car can be difficult because prices depend on several factors, including manufacturing year, kilometers driven, fuel type, transmission, ownership history, and vehicle specifications.

AutoValue AI aims to provide a data-driven price estimate that helps users make more informed decisions.

### 1.4 Product Goals
- Predict estimated used-car selling prices using machine learning.
- Identify the factors that influence car prices.
- Compare multiple regression algorithms.
- Provide a simple interface for entering car details and viewing predictions.
- Demonstrate an end-to-end machine learning workflow suitable for a student portfolio.

---

## 2. Target Users

### Primary Users
- Used-car buyers estimating a reasonable purchase price.
- Used-car sellers estimating an asking price.
- Students learning machine learning and regression.

### Secondary Users
- Data science recruiters reviewing portfolio projects.
- Automotive enthusiasts exploring vehicle pricing trends.

---

## 3. Functional Requirements

### FR-01: Car Information Input
The application must allow users to enter the following information, subject to dataset availability:

- Car brand and model
- Manufacturing year
- Kilometers driven
- Fuel type: Petrol, Diesel, CNG, etc.
- Transmission: Manual or Automatic
- Ownership history
- Engine capacity
- Mileage
- Maximum power
- Number of seats

Only features supported by the selected training dataset will be used for prediction.

### FR-02: Data Preprocessing
The system must:
- Handle missing values.
- Identify and handle invalid records.
- Detect and investigate outliers.
- Encode categorical variables.
- Scale numerical variables if required by the selected model.
- Prevent data leakage during training and evaluation.

### FR-03: Exploratory Data Analysis
The system must provide visualizations for:
- Car prices by manufacturing year.
- Price versus kilometers driven.
- Price distribution.
- Price comparison by fuel type.
- Price comparison by transmission.
- Correlations between numerical features.

### FR-04: Machine Learning Model Training
The system must support training and comparison of:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

The final model will be selected based on validation performance and appropriate error metrics.

### FR-05: Price Prediction
After the user submits the car details, the system must:
- Validate the inputs.
- Apply the same preprocessing used during training.
- Generate a predicted selling price.
- Display the estimate in Indian Rupees (₹).
- Show a helpful validation message if the input is invalid or unsupported.

### FR-06: Model Evaluation
The system must calculate:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

These metrics will be reported on held-out test data.

### FR-07: Prediction Explanation
The application should display the main factors associated with the model's prediction, using feature importance or another appropriate explanation technique where supported.

Explanations must not imply that a feature caused the predicted price.

### FR-08: Prediction History
As a future enhancement, the application may allow users to save previous predictions and compare different vehicles.

---

## 4. Non-Functional Requirements

### Performance
- Generate predictions within approximately 2 seconds under normal local-demo conditions.
- Load the model once when the application starts rather than retraining for each prediction.

### Usability
- Provide a clean, responsive interface.
- Use clear field labels and helpful input validation.
- Display the estimated price prominently.

### Reliability
- Use a consistent preprocessing and prediction pipeline.
- Handle missing model files and unexpected input errors gracefully.
- Save the trained model and preprocessing pipeline together.

### Security
- Validate user inputs.
- Do not expose API keys or other secrets in source code.
- Avoid collecting personal information that is unnecessary for predictions.

### Maintainability
- Keep data processing, model training, and UI code in separate modules.
- Document installation, training, and execution steps in the README.

---

## 5. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Data processing | NumPy, pandas |
| Visualization | Matplotlib, Seaborn |
| Machine learning | scikit-learn |
| Web interface | Streamlit |
| Model persistence | joblib |
| Development environment | VS Code or Jupyter Notebook |
| Version control | Git and GitHub |
| Dataset format | CSV |

---

## 6. Dataset Requirements

The project will initially use a publicly available used-car dataset, such as a CarDekho dataset or another dataset with suitable features and a documented source.

### Expected Input Features
- Year
- Present price or comparable vehicle-value feature, if available and appropriate
- Kilometers driven
- Fuel type
- Seller type
- Transmission
- Previous owners

### Target Variable
**Selling Price:** The historical sale price of the vehicle, represented numerically.

The exact features will depend on the selected dataset. Any feature that directly reveals the target price or would not be available when making a real prediction must be excluded to prevent target leakage.

---

## 7. User Workflow

1. The user opens AutoValue AI.
2. The application displays a short description of the tool.
3. The user enters the car's specifications.
4. The application validates the entered details.
5. The trained ML pipeline processes the input.
6. The model predicts the estimated selling price.
7. The application displays the estimate and, optionally, a prediction explanation.
8. The user can modify the inputs and generate another estimate.

---

## 8. User Interface Requirements

### Home Page
- Product name and introduction.
- Short explanation of how predictions work.
- Button or link to open the prediction form.

### Prediction Page
- Form containing the supported vehicle features.
- Predict Price button.
- Estimated price display.
- Input validation and error messages.

### Model Insights Page
- Dataset overview.
- Price distribution charts.
- Model comparison table.
- MAE, RMSE, and R² results.
- Feature importance visualization, where supported.

---

## 9. Model Success Criteria

The model must outperform a simple baseline predictor, such as predicting the median selling price from the training data, on an appropriate held-out test set.

The following criteria will be used:

- MAE should be low relative to the price range of the dataset.
- RMSE should be reviewed to identify large prediction errors.
- R² should be reported to describe the model's explanatory performance on the test set.
- The selected model should perform consistently on unseen examples.

No fixed accuracy percentage is guaranteed in advance. Performance targets will be finalized after dataset inspection and baseline evaluation.

---

## 10. Project Scope

### Minimum Viable Product (MVP)
- Public used-car dataset.
- Data cleaning and exploratory analysis.
- At least two trained regression models.
- Model comparison using MAE, RMSE, and R².
- Saved model and preprocessing pipeline.
- Streamlit prediction form.
- Price estimate in Indian Rupees.
- GitHub repository with documentation.

### Future Enhancements
- Compare predicted prices across multiple cars.
- Add a broader collection of car brands and models.
- Explain predictions using SHAP or another suitable method.
- Integrate a database to save prediction history.
- Deploy the application publicly.
- Add market trend analysis if reliable, relevant data becomes available.

Live market pricing, user accounts, and automated data collection are outside the initial MVP.

---

## 11. Development Milestones

### Phase 1: Data Collection
Find a suitable dataset, inspect its columns, and define the prediction target.

### Phase 2: Data Analysis
Clean the dataset, perform exploratory analysis, and visualize relationships between features and prices.

### Phase 3: Model Development
Build preprocessing pipelines, train regression models, and evaluate their performance.

### Phase 4: Application Development
Build the Streamlit interface and integrate the saved model.

### Phase 5: Testing
Test input validation, prediction consistency, model loading, and error handling.

### Phase 6: Documentation and Deployment
Create a README, add screenshots, document the metrics, and publish the code on GitHub.

---

## 12. Testing Requirements

The application must be tested for:

- Valid vehicle inputs.
- Missing or invalid numeric values.
- Unsupported categories.
- Manufacturing years outside the supported range.
- Very high or low kilometer readings.
- Missing or corrupted model files.
- Consistency between notebook predictions and web application predictions.

Predictions should be treated as estimates rather than guaranteed resale values.

---

## 13. Risks and Limitations

- Dataset prices may not reflect current market conditions.
- Some brands or vehicle categories may have insufficient training examples.
- Missing or unreliable features may reduce model quality.
- The model may perform poorly on cars outside the training distribution.
- Historical selling prices may vary by location, vehicle condition, and market demand.

The interface must communicate that actual resale prices depend on factors not necessarily captured by the model.

---

## 14. Final Deliverables

1. Dataset source and data dictionary.
2. Exploratory data analysis notebook.
3. Data preprocessing and model training scripts.
4. Trained model and preprocessing pipeline.
5. Streamlit web application.
6. Model evaluation report.
7. GitHub repository with setup instructions.
8. Screenshots or a demonstration video.

---

## 15. Definition of Done

The project will be considered complete when:

- A documented dataset has been selected.
- Data preprocessing and model training run successfully.
- At least two regression models have been evaluated.
- The chosen model beats the baseline on the held-out test set.
- The web application produces predictions from valid inputs.
- Model evaluation metrics are documented.
- Basic application tests pass.
- The GitHub README includes setup and execution instructions.

**Final Outcome:** A complete, documented, end-to-end machine learning application that estimates used-car prices and demonstrates practical skills in Python, data analysis, regression, model evaluation, and deployment.
