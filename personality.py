import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

data = pd.read_csv("/home/dustoph/person-app/personality_dataset.csv")
df = pd.DataFrame(data)
df.dropna(inplace= True)

df['Personality'] = df['Personality'].map({'Introvert': 0, 'Extrovert': 1})
df['Stage_fear'] = df['Stage_fear'].map({'No': 0, 'Yes': 1})
df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'No': 0, 'Yes': 1})
y = df['Personality']
X = df.drop('Personality', axis= 1)
features= X
class_names = [str(c) for c in y.unique()]
print(df.head(5))

print(df['Personality'].value_counts()[0])
print(df.head(5))

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=1)
# print(X_train)
print(y_train.unique())

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test,y_pred))

clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded = True, special_characters=True, feature_names=X.columns, class_names=class_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Personality.png')
Image(graph.create_png())


import streamlit as st
# import pandas as pd
import joblib  # to load the trained model
mdl = joblib.dump(clf, 'personality_model.pkl')
# Load trained model
model = joblib.load("personality_model.pkl")  # save your trained model using joblib.dump(model, 'personality_model.pkl')

st.title("Personality Type Predictor")
st.write("Answer the following questions to know your personality type.")

# Ask for user input
age = st.slider("Age", 10, 100, 25)
time_spent_alone = st.slider("Alone time in hrs", 1, 10 ,2)
stage_fear = st.radio("Do you have stage fear?", ["Yes", "No"])
social_event_attendance = st.slider("are you present in social events?",1, 10, 2)
going_outside = st.slider("how frequently do you go outside",1,10,2)
drained = st.radio("Do you feel drained after socializing?", ["Yes", "No"])
friends_circle_size = st.slider("How many friends do you have?", 0, 100, 10)
# hobbies = st.slider("How many hobbies do you actively pursue?", 0, 20, 2)
post_frequency = st.slider("How often do you post on social media?",1,10,2)
# Map inputs
stage_fear_val = 1 if stage_fear.lower() == "yes" else 0
drained_val = 1 if drained.lower() == "yes" else 0

# Create input for prediction
input_df = pd.DataFrame([{
    # "Age": age,    ####
    "Time_spent_Alone": time_spent_alone,
    "Stage_fear": stage_fear_val,
    "Social_event_attendance": social_event_attendance,
    "Going_outside": going_outside,
    "Drained_after_socializing": drained_val,
    "Friends_circle_size": friends_circle_size,
    "Post_frequency": post_frequency,
    # "Hobbies_count": hobbies    #####
}])

# Make prediction
if st.button("Predict Personality"):
    prediction = model.predict(input_df)[0]
    result = "Extrovert" if prediction == 1 else "Introvert"
    st.success(f"🎉 You are likely an **{result}**!")


joblib.dump(clf, "personality_model.pkl")

