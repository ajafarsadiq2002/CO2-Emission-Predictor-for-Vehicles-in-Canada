# CO2 Emission Predictor for Vehicles in Canada
# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit.components.v1 as components


# Step 2: Load and Explore Data
def load_data():
    st.markdown("<h1 style='font-size: 30px;'>📥 Data Uploading</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded and loaded successfully!")
            return data
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            return None
    else:
        st.info("👆 Please upload a CSV file to proceed.")
        return None


def explore_data(data):
    st.markdown("<h1 style='font-size: 30px;'>📥 Data Exploration</h1>", unsafe_allow_html=True)
    with st.expander("🔍 Dataset Overview"):
        st.write("#### Preview Dataset Rows")

        # row_count = st.slider("Select number of rows to display", min_value=1, max_value=len(data), value=5, step=1)
        # st.dataframe(data.head(row_count))

        
        row_count = st.number_input(
            label="Enter number of rows to display",
            min_value=1,
            max_value=len(data),
            value=5,
            step=1,
            format="%d"  # forces integer formatting
        )

        st.dataframe(data.head(int(row_count)))

        st.write("#### 📋 Structured Dataset Info")
        info_table = pd.DataFrame({
            "Column": data.columns,
            "Non-Null Count": data.notnull().sum().values,
            "Dtype": [str(dtype) for dtype in data.dtypes.values]
        })
        st.dataframe(info_table)

        # Extra info
        st.markdown("#### ℹ️ Additional Info")
        dtypes_summary = data.dtypes.value_counts()
        for dtype, count in dtypes_summary.items():
            st.write(f"**{dtype}** columns: {count}")
        
        mem_usage = data.memory_usage(deep=True).sum() / 1024  # KB
        st.write(f"**Total memory usage:** {mem_usage:.1f} KB")

        st.write("#### Summary Statistics:")
        st.dataframe(data.describe())

        st.write("#### Missing Values:")
        missing_df = data.isnull().sum().reset_index()
        missing_df.columns = ["Column", "Missing Values"]
        st.dataframe(missing_df)

    with st.expander("🔢 Correlation Matrix Heatmap"):
        # Encode categorical features
        for col in ['Make', 'Model', 'Vehicle Class', 'Fuel Type', 'Transmission']:
            frequency = data[col].value_counts(normalize=True)
            data[col] = data[col].map(frequency)

        # Keep only numeric columns
        data = data.select_dtypes(include=['float64', 'int64'])
        data = data.dropna()
        data = data.loc[:, (data != data.iloc[0]).any()]

        # Correlation heatmap
        correlation_matrix = data.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)

        # Add interpretation/observation text
        st.markdown("""
        ---

        ### 📌 Correlation Matrix Interpretation

        Based on the context of the dataset, here are some likely observations:

        #### 🔴 Correlation Indicator
        - **Darker red** indicates strong **positive** correlations.
        - **Darker blue** indicates strong **negative** correlations.
        - **Numbers** inside the heatmap show the **correlation coefficients** (e.g., `0.92`, `-0.91`).

        #### 📈 Positive Correlations
        - **Engine Size (L)** and **Cylinders** → CO2 Emissions (g/km):
        - Larger engines generally emit more CO2.
        - **Fuel Consumption Comb (L/100 km)**, **Hwy**, and **City**:
        - Higher fuel consumption is directly linked to higher emissions.

        #### 📉 Negative Correlations
        - **Fuel Consumption Comb (mpg)** → CO2 Emissions (g/km):
        - Higher fuel efficiency (more mpg) means lower emissions.

        #### 🔁 Redundancies
        - Fuel Consumption City, Hwy, and Combined are **highly correlated** with each other.
        - These may be **redundant** features for modeling.

        #### ⚪ Weak Correlations
        - Encoded categorical variables like **Make**, **Model**, **Vehicle Class**, **Fuel Type**, and **Transmission** show weak correlations with CO2 Emissions.
        """)


def preprocess_data(data):
    st.markdown("<h1 style='font-size: 30px;'> 🧹 Data Preprocessing</h1>", unsafe_allow_html=True)

    with st.expander("🔍 Why are we dropping 'Make', 'Model', 'Vehicle Class', 'Fuel Type' and 'Transmission'?"):
        st.markdown("""
We drop these columns for the following reasons:

- **`Make` and `Model`**:  
  Have **too many unique values**, which makes them **sparse** and difficult to encode meaningfully.
  Shows **weak correlation** with CO2 emissions and is likely **not a strong predictor** in this context.

- **`Vehicle Class`**:  
  May be **redundant** with other numerical indicators like **Engine Size**, **Fuel Consumption**, etc.
  Shows **weak correlation** with CO2 emissions and is likely **not a strong predictor** in this context.

- **`Transmission`**:  
  Shows **weak correlation** with CO2 emissions and is likely **not a strong predictor** in this context.

- **`Fuel Type`**:
  Shows **weak correlation** with CO2 emissions and is likely **not a strong predictor** in this context.

> ✅ We use `errors="ignore"` so that the code **doesn't break** if any of these columns are already missing.
        """)

    data = data.drop(['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type'], axis=1, errors='ignore')

    with st.expander("📘 Why are we splitting the dataset into training and testing sets?"):
        st.markdown("""
We split the dataset into:

- **Features (X):** All input variables includes Engine Size(L), Cylinders, Fuel Consumption City (L/100 km), Fuel Consumption Hwy (L/100 km), Fuel Consumption Comb (L/100 km), Fuel Consumption Comb (mpg) (excluding the target).
- **Target (y):** The output we want to predict → `CO2 Emissions(g/km)`

Then we further split them into:

- **Training set (80%)** – used to teach the model  
- **Testing set (20%)** – used to evaluate how well the model generalizes

This helps:
- Prevent **overfitting**
- Give a **reliable estimate** of model performance on unseen data

> ✨ Think of it like teaching a student using 80% of the textbook, and testing them with the remaining 20%.
        """)

    # Actual split
    X = data.drop('CO2 Emissions(g/km)', axis=1)
    y = data['CO2 Emissions(g/km)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Display split sizes
    with st.expander("📊 Dataset Split Summary"):
        st.markdown(f"""
**Total Samples:** {len(data)}  
- 🏋️ **Training Features (X_train):** {X_train.shape[0]} rows, {X_train.shape[1]} columns  
- 🧪 **Testing Features (X_test):** {X_test.shape[0]} rows, {X_test.shape[1]} columns  
- 🎯 **Training Labels (y_train):** {y_train.shape[0]}  
- 🎯 **Testing Labels (y_test):** {y_test.shape[0]}
        """)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    st.info("🤖 Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    st.subheader("📊 Model Evaluation Metrics")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("MSE", f"{mse:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("R² Score", f"{r2:.2f}")

    return mse, rmse, r2

# Streamlit App Entry Point
if __name__ == "__main__":
    st.set_page_config(page_title="CO2 Emission Predictor", layout="centered")
    
    st.markdown("<h1 style='text-align: center; font-size: 40px; color: red;'>🚗 CO2 Emission Predictor for Vehicles in Canada</h1>", unsafe_allow_html=True)

    st.markdown("Machine Learning Engineers : Allabaksh S[G24AIT2164] and Jafar Sadiq[G24AIT2152]")

    st.markdown("This app predicts the **CO2 Emissions (g/km)** from vehicle specifications using a trained machine learning model.")

    # Step 1: Load and Explore Data
    data = load_data()
    if data is not None:
        explore_data(data)

        # Step 2: Preprocess Data
        X_train, X_test, y_train, y_test = preprocess_data(data)
        feature_names = X_train.columns.tolist()

        # Step 3: Train Model
        model = train_model(X_train, y_train)

        # Step 4: Evaluate Model
        evaluate_model(model, X_test, y_test)

        # Step 5: Prediction Input UI
        st.subheader("🧮 Enter Vehicle Details to Predict CO2 Emissions")
        st.markdown("The allowed threshold for CO2 emissions is **260 g/km**.")

        # Get mode values from the dataset
        engine_size_mode = float(data['Engine Size(L)'].mode()[0])
        cylinders_mode = int(data['Cylinders'].mode()[0])
        fuel_city_mode = float(data['Fuel Consumption City (L/100 km)'].mode()[0])
        fuel_hwy_mode = float(data['Fuel Consumption Hwy (L/100 km)'].mode()[0])
        fuel_comb_l_mode = float(data['Fuel Consumption Comb (L/100 km)'].mode()[0])
        fuel_comb_mpg_mode = float(data['Fuel Consumption Comb (mpg)'].mode()[0])

        col1, col2 = st.columns(2)
        with col1:
            engine_size = st.number_input("Engine Size (L)", min_value=0.0, max_value=10.0, value=engine_size_mode)
            cylinders = st.number_input("Cylinders", min_value=2, max_value=16, value=cylinders_mode)
            fuel_city = st.number_input("Fuel Consumption City (L/100 km)", min_value=0.0, max_value=50.0, value=fuel_city_mode)

        with col2:
            fuel_hwy = st.number_input("Fuel Consumption Hwy (L/100 km)", min_value=0.0, max_value=50.0, value=fuel_hwy_mode)
            fuel_comb_l = st.number_input("Fuel Consumption Combined (L/100 km)", min_value=0.0, max_value=50.0, value=fuel_comb_l_mode)
            fuel_comb_mpg = st.number_input("Fuel Consumption Combined (mpg)", min_value=0.0, max_value=100.0, value=fuel_comb_mpg_mode)
            
        if st.button("🖼️ Show Test Cases"):
                st.image("q4_test_data.jpg", caption="Please refer this to validate the prediction of the model", use_container_width=True)    
        if st.button("🔍 Predict CO2 Emissions"):

            input_data = {
                'Engine Size(L)': engine_size,
                'Cylinders': cylinders,
                'Fuel Consumption City (L/100 km)': fuel_city,
                'Fuel Consumption Hwy (L/100 km)': fuel_hwy,
                'Fuel Consumption Comb (L/100 km)': fuel_comb_l,
                'Fuel Consumption Comb (mpg)': fuel_comb_mpg
            }

            input_df = pd.DataFrame([input_data])
            input_df = input_df.reindex(columns=feature_names, fill_value=0)

            prediction = model.predict(input_df)[0]

            # Define threshold for "high emission"
            co2_limit = 260  # Example upper limit (adjust if needed)

            st.success(f"🚘 Predicted CO2 Emissions: **{prediction:.2f} g/km**")

            if prediction > co2_limit:
                st.warning(f"⚠️ This value exceeds the typical upper limit for modern vehicles in Canada (~{co2_limit} g/km).")

                st.markdown("""
                **Vehicles emitting more than `260 g/km` are considered _high-polluting_.**

                Here are some suggestions to reduce emissions:

                - ✅ Use more **fuel-efficient engines**
                - ⚡ Consider **hybrid or electric alternatives**
                - 🧰 Keep up with **regular vehicle maintenance**
                """)
            else:
                st.info("✅ This predicted emission value is within a typical range for compliant vehicles.")

            # 📄 Markdown Summary
            st.markdown("---")
            st.markdown("### 🧾 Prediction Summary")
            summary = f"""Vehicle Emission Prediction Summary
    -----------------------------------
    Engine Size: {engine_size} L
    Cylinders: {cylinders}
    Fuel Consumption (City): {fuel_city} L/100km
    Fuel Consumption (Hwy): {fuel_hwy} L/100km
    Fuel Consumption (Combined): {fuel_comb_l} L/100km
    Fuel Efficiency (mpg): {fuel_comb_mpg}
    Predicted CO₂ Emissions: {prediction:.2f} g/km
    """
            st.download_button("📄 Download Summary", summary, file_name="co2_summary.txt")


