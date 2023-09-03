# Import necessary libraries
import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Configure Streamlit page settings
st.set_page_config(
    page_title="Advanced Laptop Price Predictor",
    page_icon=":💻:",
    layout="centered",
)

# Load the data and machine learning model
data = pd.read_csv("df_final.csv")
with open("laptoppricepredictor.pkl", "rb") as file:
    model = pickle.load(file)

# Create the main title
st.title("Advanced Laptop Price Predictor")

# Dropdowns and input fields for laptop features
Company = st.selectbox("Brand", sorted(data['Company'].unique()),)
st.markdown(f"selected brand: :green[{Company}]",)

Type = st.selectbox("Type", sorted(data['Type'].unique()),)
st.markdown(f"selected Type: :green[{Type}]",)

OpSys = st.selectbox("Operating system", sorted(data['OpSys'].unique()),)
st.markdown(f"selected Operating system: :green[{OpSys}]",)

CPU = st.selectbox("CPU", sorted(data['CPU'].unique()),)
st.markdown(f"selected CPU: :green[{CPU}]",)

GPU = st.selectbox("GPU", sorted(data['GPU'].unique()),)
st.markdown(f"selected GPU: :green[{GPU}]",)

ClockSpeed = st.number_input("ClockSpeed (in GHz)", min_value=1.0, max_value=3.0, help="ranges 1GHz to 3GHz")
st.markdown(f"selected ClockSpeed: :green[{ClockSpeed} GHz]",)

Ram = st.selectbox("Ram (in GB)", sorted(data['Ram'].unique()),)
st.markdown(f"selected Ram: :green[{Ram} GB]",)

HDD = st.selectbox("HDD (in TB)", sorted(data['HDD'].unique()),)
st.markdown(f"selected HDD: :green[{HDD} TB]",)

SSD = st.selectbox("SSD (in TB)", sorted(data['SSD'].unique()),)
st.markdown(f"selected SSD: :green[{SSD} TB]",)

Weight = st.number_input("Weight", min_value=1.0, max_value=4.0, help="ranges 1kg to 4kg")
st.markdown(f"selected Weight: :green[{Weight} kg]",)

Touchscreen = st.selectbox("Touch screen", ["Yes","No"],)
st.markdown(f"selected Touch screen: :green[{Touchscreen}]",)

Display = st.selectbox("Display (in GB)", sorted(data['Display'].unique()),)
st.markdown(f"selected Display: :green[{Display}]",)

screen_size = st.number_input('Screen Size (in inches)', min_value=11.0, max_value=17.0, help="Ranges 11 inches to 17 inches")
st.markdown(f"selected Display: :green[{screen_size} inches]",)

resolution = st.selectbox('Screen Resolution (W x H)', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
st.markdown(f"selected Display: :green[{resolution}]",)

# creating buttons
col1, col2, col3 = st.columns(3)
confirm = col1.button("Predict price",help = "click to predict the price", type="primary")
compare = col2.button("Compare",help = "Compare laptops")
clear = col3.button("Clear",help = "Clear comparison")

# dropdowns for comparison
parameter_1 = st.selectbox('X axis', data.columns)
parameter_2 = st.selectbox('Y axis', data.columns)

# Convert Touchscreen to binary
if Touchscreen == 'Yes':
    Touchscreen = 1
else:
    Touchscreen = 0

# Calculate PPI based on screen resolution and size
X_resolution = int(resolution.split('x')[0])
Y_resolution = int(resolution.split('x')[1])
ppi = np.sqrt((X_resolution**2) + (Y_resolution**2)) / (screen_size)

# Gather details in a dictionary to convert it to dataframe
details = [Company, Type, OpSys, CPU, GPU, Ram, Weight, Touchscreen, ClockSpeed, HDD, SSD, ppi, Display]
df = {feature: value for feature, value in zip(data.columns, details)}

# Predict laptop price using the machine learning model
price = int(np.exp(model.predict(pd.DataFrame(df, index=[0]))))

# Create a plot for comparison
fig, ax = plt.subplots()

# Initialize or update the comparison table
if 'comparison_state' not in st.session_state:
    st.session_state.comparison_state = pd.DataFrame(columns=data.columns)
    st.session_state.index = 0

# Display predicted price if "Predict price" button is clicked
if confirm:
    st.markdown(f"***Price of Laptop = :green[{price}]***")

# Add laptop to comparison table if "Compare" button is clicked
if compare:
    st.session_state.comparison_state = st.session_state.comparison_state.append(pd.Series(df), ignore_index=True)
    st.session_state.comparison_state.at[st.session_state.index, 'Price'] = price
    st.write("Laptop added to comparison.")
    st.session_state.index += 1
    
    # Create a bar plot for comparison
    sns.barplot(data=st.session_state.comparison_state, x=parameter_1, y='Price', hue=parameter_2)
    st.pyplot(fig)

# Clear the comparison table if "Clear" button is clicked
if clear:
    st.session_state.comparison_state = pd.DataFrame(columns=data.columns)
    st.write("Comparison table cleared.")
    st.session_state.index = 0

# Display the comparison table
if not st.session_state.comparison_state.empty:
    st.write("Comparison Table:")
    st.write(st.session_state.comparison_state)