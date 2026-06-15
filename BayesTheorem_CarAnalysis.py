import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

# Creatingcreating a Python dictionary named data that stores three lists of values, 
# each representing a feature of cars 
data = {
    "Weight": [1000, 1200, 1500, 1800, 2000],
    "Engine":[1.2, 1.5, 2.0, 2.4, 3.0],
    "Fuel":[20, 18, 14, 12, 10]
}


# Making a dataframe of the data using pandas library
df = pd.DataFrame(data)
print(df)

# Plotting the graph of weight vs engine
plt.figure(figsize= (6,5))
plt.scatter(df["Weight"],df["Engine"])
plt.xlabel("Weight (kg)")
plt.ylabel("Engine")
plt.title("Weight vs Engine")
plt.grid()
plt.show()

# Calculating mean and standard deviation
mn = df["Fuel"].mean()
standev = df["Fuel"].std()

print("Mean =",mn)
print("Standard Deviation =",standev)

# linspace is line spacing used to plot a line by taking values between 5 and 25
x = np.linspace(5, 25)
# norm stands for normal distrubution
# pdf stands for probability distribution function
y = norm.pdf(x, mn, standev)


# Plotting helps predict model performance by just looking at it
plt.figure(figsize=(6,5))
plt.plot(x,y, marker = 'o')
# plt.axvline Adds a vertical line spanning the whole or fraction of the Axes.
plt.axvline(mn, linestyle="--")
plt.title("Normal Distribution")
plt.xlabel("Fuel Consumption")
plt.ylabel("Probablity Density")
plt.grid()
plt.show()


# Now we are taking a case with following criteria and check conditional probability
A = df["Fuel"] < 12
print("A=",A,"------")

B = (df["Weight"] > 1500) & (df["Engine"]> 2.0)
print("B=",B)


# Gives the length of the dataframe
N = len(df)
print(N)


# Calculates the probability of event A 
P_A = sum(A)/N 

# Calculates the probability of event B
P_B = sum(B)/N

# Conditional probability: the chance of B happening given that A has already occurred
P_B_given_A = sum(A & B) / sum(A)

# This is Bayes’ theorem. It calculates the probability of A given B, using the relationship between conditional probabilities.
P_A_given_B = (P_B_given_A * P_A)/P_B


print("\n--- BAYES RESULTS ---")
print("P(A) =", P_A)
print("P(B) =", P_B)
print("P(B|A) =", P_B_given_A)
print("P(A|B) =", P_A_given_B)


