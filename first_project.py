import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data = pd.DataFrame({
    "HoursStudied" : [1 , 4 , 5 , 6 ,2],
    "Score" : [38 , 68 , 74 , 86 , 50]
})

X = data[["HoursStudied"]]
y = data["Score"]

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

model = LinearRegression()

model.fit(X_train , y_train)

y_pred = model.predict(X_test)

new_student = model.predict(pd.DataFrame({"HoursStudied" : [7]}))

print("MSE", mean_squared_error(y_test , y_pred))
print("New Study Score" , new_student)

# Plot data points
plt.scatter(X, y, color="blue", label="Actual Data")

# Regression line
plt.plot(X, model.predict(X), color="red", label="Regression Line")

# New prediction point
plt.scatter(7, new_student[0], color="green", s=100, label="Prediction (7 hrs study)")

# Labels and title
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Hours Studied vs Score")
plt.legend()
plt.show()
