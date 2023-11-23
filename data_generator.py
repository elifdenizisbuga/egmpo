import numpy as np

# Define the number of samples
num_samples = 1000

# Generate synthetic data for the features (input)
age = np.random.randint(25, 65, num_samples)  # Random ages between 25 and 65
education_level = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], num_samples)
salary = np.random.uniform(30000, 150000, num_samples)  # Random salaries between $30,000 and $150,000
risk_level = np.random.choice(['Low', 'Medium', 'High'], num_samples)

# Combine the features into a feature matrix
X = np.column_stack((age, education_level, salary, risk_level))

# Generate synthetic data for the target variables (weights of funds)
num_funds = 20
weights = np.random.rand(num_samples, num_funds)

# Now, 'X' contains your input features, and 'weights' contains the target weight allocations for the funds.
print(weights)
