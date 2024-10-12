
# Step 1 import library
import pandas as pd


# In[2]:


#step import data
df = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Salary%20Data.csv')


# In[41]:


#displaying the dataset
df


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.columns


# In[8]:


df.shape


# In[28]:


# the iloc function is used for index-based selection of rows and columns
x = df.iloc[ :, 1:]
y = df.iloc[ :, 1:]


# In[29]:


x.shape


# In[30]:


#trains test split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 


# In[31]:


model = LinearRegression()


# In[32]:


model.fit(x_train, y_train)


# In[33]:


y_pred = model.predict(x_test)
y_pred #The model predicts salary values for the x_test data.


# In[34]:


y_train


# In[35]:


from sklearn.metrics import mean_squared_error, r2_score


# In[36]:


print('Mean Squared Error:', mse)


# In[ ]:





# In[39]:


import pickle
import numpy as np
import streamlit as st
# Load the trained model
def load_model():
    with open('salary_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model
# Main function for the Streamlit app
def main():
    st.title('Experience-Based Salary Prediction')

    # Input for years of experience
    experience = st.number_input('Enter years of experience:', min_value=0.0, max_value=50.0, value=1.0, step=0.5)

    # Predict salary based on user input
    if st.button('Predict Salary'):
        model = load_model()  # Load the model
        input_data = np.array([[experience]])  # Reshape the input for prediction
        predicted_salary_usd = model.predict(input_data)  # Predict in USD

        # Convert numpy array to float (USD)
        salary_usd = predicted_salary_usd[0].item()  # Convert to Python float

        # Conversion rate (1 USD = 83 INR)
        conversion_rate = 27010+9343*salary_usd
        salary_inr = salary_usd * conversion_rate  # Convert USD to INR

        # Display the predicted salary directly in INR
        st.write(f'Predicted Salary: ₹{salary_inr:.2f} INR')

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




