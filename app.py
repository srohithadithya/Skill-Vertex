from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC

app = Flask(__name__)

# Load and train the model
def load_and_train_model():
    # Load the dataset
    data = pd.read_csv('Sample_Employee_Data.csv')  # Update with your CSV file path

    # Preprocess categorical variables
    label_encoders = {}
    categorical_cols = ['BusinessTravel', 'Department', 'JobRole', 'Gender']
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Map binary columns
    data['PastEmployee'] = data['PastEmployee'].map({'Yes': 1, 'No': 0})

    # Features and target variable
    X = data.drop('PastEmployee', axis=1)
    y = data['PastEmployee']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=100)
    X_train, y_train = SMOTE(random_state=100).fit_resample(X_train, y_train)

    # Train the SVM model
    model = SVC(kernel='linear', random_state=0)
    model.fit(X_train, y_train)

    return model, scaler, label_encoders

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load model and preprocessors
    model, scaler, label_encoders = load_and_train_model()

    # Get input data from the form
    age = float(request.form['age'])
    gender = request.form['gender']
    business_travel = request.form['business_travel']
    department = request.form['department']
    job_role = request.form['job_role']
    num_companies_worked = float(request.form['num_companies_worked'])
    over_time = request.form['over_time']
    past_employee = request.form['past_employee']

    # Encode categorical values
    gender_encoded = 1 if gender == 'Female' else 0
    business_travel_encoded = label_encoders['BusinessTravel'].transform([business_travel])[0]
    department_encoded = label_encoders['Department'].transform([department])[0]
    job_role_encoded = label_encoders['JobRole'].transform([job_role])[0]
    over_time_encoded = 1 if over_time == 'Yes' else 0
    past_employee_encoded = 1 if past_employee == 'Yes' else 0

    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_encoded],
        'BusinessTravel': [business_travel_encoded],
        'Department': [department_encoded],
        'JobRole': [job_role_encoded],
        'NumCompaniesWorked': [num_companies_worked],
        'OverTime': [over_time_encoded],
        'PastEmployee': [past_employee_encoded]
    })

    # Scale input data
    preprocessed_data = scaler.transform(input_data)

    # Make a prediction
    prediction = model.predict(preprocessed_data)

    return render_template('index.html', prediction='Yes' if prediction[0] == 1 else 'No')

if __name__ == '__main__':
    app.run(debug=True)