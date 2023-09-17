import streamlit as st
import pickle
import numpy as np
import sklearn
# Load the pre-trained linear regression model
def load_model(model_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Load the pre-trained linear regression model (replace 'your_model_path.pkl' with the actual path)
model_path = 'lr_model.pkl'
model = load_model(model_path)

# Define a function to predict using the model
def predict_employee_performance(employee_id, gender, age, relationship_status, hometown, unit,
                                 decision_skill_possess, time_of_service, time_since_promotion,
                                 growth_rate, travel_rate, post_level, pay_scale,
                                 compensation_and_benefits, work_life_balance):
    # Preprocess the input
    input_data = np.array([[employee_id, gender, age, relationship_status, hometown, unit,
                            decision_skill_possess, time_of_service, time_since_promotion,
                            growth_rate, travel_rate, post_level, pay_scale,
                            compensation_and_benefits, work_life_balance]])

    # Predict using the model
    predicted_performance = model.predict(input_data)

    return predicted_performance[0]  # Assuming the model predicts a single value

# Streamlit app
st.title("Attrition Rate Prediction")

# Input fields
employee_id = st.number_input("Employee ID")
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", min_value=21, max_value=60, value=25)
relationship_status =st.selectbox("Relationship status",["Married","Single"])
Hometown = st.selectbox("Hometown",['Franklin', 'Springfield', 'Clinton', 'Lebanon', 'Washington'])
Decision_skill_possess = st.selectbox("Decision Skill Possess",['Conceptual', 'Directive', 'Analytical', 'Behavioral'])
Time_of_service = st.number_input("Time of service")
Time_since_promotion = st.number_input("Time since promotion")
growth_rate = st.number_input("Growth rate(out of 100)")
Travel_Rate = st.number_input("Travel Rate")
Post_Level = st.selectbox("Post Level",[1,2,3,4,5])
Pay_Scale = st.selectbox("Pay Scale",[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0])
Compensation_and_Benefits = st.selectbox("Compensation and Benefits",['Type0','Type1','Type2','Type3','Type4'])
Work_Life_balance = st.selectbox("Work Life balance",[0.0,1.0,2.0,3.0,4.0,5.0])
Unit = st.selectbox("Unit",['R&D', 'IT', 'Sales', 'Marketing', 'Purchasing', 'Operarions','Human Resource Management', 'Logistics', 'Accounting and Finance','Security', 'Quality', 'Production'])

# Add more input fields for the remaining features
# Convert gender to numeric (0 or 1)
gender = 0 if gender == "Male" else 1
relationship_status = 1 if relationship_status=="Married" else 0
age = float(age)
Time_of_service = float(Time_of_service)


if Hometown=="Clinton":
    Hometown=0
elif Hometown=="Franklin":
    Hometown=1
elif Hometown=="Lebanon":
    Hometown=2
elif Hometown=="Springfield":
    Hometown=3
else:
    Hometown=4


if Decision_skill_possess=="Analytical":
    Decision_skill_possess = 0
elif Decision_skill_possess=="Behavioral":
    Decision_skill_possess=1
elif Decision_skill_possess=="Conceptual":
    Decision_skill_possess=2
else:
    Decision_skill_possess=3


if Compensation_and_Benefits=="Type0":
    Compensation_and_Benefits=0
elif Compensation_and_Benefits=="Type1":
    Compensation_and_Benefits=1
elif Compensation_and_Benefits=="Type2":
    Compensation_and_Benefits=2
elif Compensation_and_Benefits=="Type3":
    Compensation_and_Benefits=3
else:
    Compensation_and_Benefits = 4


if Unit== "Accounting and Finance":
    Unit = 0
elif Unit=="Human Resource Management":
    Unit = 1
elif Unit=="IT":
    Unit=2
elif Unit=="Logistics":
    Unit = 3
elif Unit=="Marketing":
    Unit = 4
elif Unit=="Operations":
    Unit=5
elif Unit=="Purchasing":
    Unit = 6
elif Unit=="Quality":
    Unit = 7
elif Unit=="R&D":
    Unit=8
elif Unit=="Sales":
    Unit=9
else:
    Unit=10


# Predict button
if st.button("Predict"):
    # Call the prediction function with input values
    predicted_value = predict_employee_performance(
        employee_id, gender, age,relationship_status,Hometown,Unit,Decision_skill_possess,Time_of_service, Time_since_promotion,growth_rate, Travel_Rate, Post_Level, Pay_Scale,Compensation_and_Benefits, Work_Life_balance# Add more input values here
    )
    st.write(f"Predicted Performance: {predicted_value:.2f}")
