import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class SailboatSteeringModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

# Generate synthetic data for training (wind speed, wind direction, steering angle)
def generate_data(num_samples=1000):
    X = [[random.uniform(0, 30), random.uniform(0, 360)] for _ in range(num_samples)]
    y = [random.uniform(-30, 30) for _ in range(num_samples)]
    return X, y

# Example usage
X_data, y_data = generate_data()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Create and train the sailboat steering model
model = SailboatSteeringModel()
model.train(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
