import pandas as pd
import numpy as np
import pickle
import streamlit as smt
from PIL import Image
import sklearn
from matplotlib import pyplot as plt
import seaborn as sns
#from sklearn import datasets
  
# loading in the model to predict on the data
smt.set_page_config(layout="wide")
pickle_in = open('model.pkl', 'rb')
model = pickle.load(pickle_in)
#model=pd.read_csv("C:\Users\BRAVE BARAKA\Breast cancer prediction\data.csv")
smt.title("Covid-19 deaths Prediction Model")
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
html_temp = """
    <div style ="background-color:blue ;padding:13px">
    <h1 style ="color:black;text-align:center;">Covid-19 DeathsPrediction ML App by BRAVE BARAKA</h1>
    </div>
    """
smt.write("This application will be used to predict the number of people succembed to Covid-19")
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
smt.markdown(html_temp, unsafe_allow_html = True)
df=pd.read_csv("corona.csv")
smt.title('Welcome all')
smt.write("Let's view our dataset first")
smt.dataframe(data=df)
df=pd.DataFrame(df,columns=['Weekly Cases', 'Weekly Deaths', 'Total Vaccinations', 'People Vaccinated',
       'People Fully Vaccinated', 'Total Boosters', 'Daily Vaccinations', 'Daily People Vaccinated', "Next Week's Deaths"])
smt.title("Analysis")
smt.write("Below is the analysis of the dataset")
smt.header("Subplots")
fig=plt.figure()
x=df["Weekly Cases"]
y=df["Next Week's Deaths"]
plt.scatter(x,y,color="blue")
smt.pyplot(fig)
#smt.write(sns.displot(x="Weekly Cases",kde=True,data=df,bins=5))

#smt.write(sns.scatterplot(x="Weekly Cases",y="Next Week's Deaths",data=df,hue="Total Vaccinations"))
#smt.header("Altair chart")
#smt.altair_chart(df)

    
def prediction(Weekly_Cases,Weekly_Deaths,Total_Vaccinations,People_Vaccinated,
    People_Fully_Vaccinated,Total_Boosters,Daily_Vaccinations,Daily_People_Vaccinated):  
   
    prediction=model.predict([[Weekly_Cases,Weekly_Deaths,Total_Vaccinations,People_Vaccinated,
    People_Fully_Vaccinated,Total_Boosters,Daily_Vaccinations,Daily_People_Vaccinated]])
    print(prediction)
    return prediction
def main():
    smt.sidebar.header("Choose your inputs")
    Weekly_Cases=smt.sidebar.number_input("Weekly_Cases")
    Weekly_Deaths=smt.sidebar.number_input("Weekly_Deaths")
    Total_Vaccinations=smt.sidebar.number_input("Total_Vaccinations")
    People_Vaccinated=smt.sidebar.number_input("People_Vaccinated")
    People_Fully_Vaccinated=smt.sidebar.number_input("People_Fully_Vaccinated")
    Total_Boosters=smt.sidebar.number_input("Total_Boosters")
    Daily_Vaccinations=smt.sidebar.number_input("Daily_Vaccinations")
    Daily_People_Vaccinated =smt.sidebar.number_input("Daily_People_Vaccinated")
    
    
    user_data={'Weekly Cases':Weekly_Cases, 'Weekly Deaths':Weekly_Deaths, 'Total Vaccinations':Total_Vaccinations, 'People Vaccinated':People_Vaccinated, 'People Fully Vaccinated':People_Fully_Vaccinated, 'Total Boosters':Total_Boosters, 'Daily Vaccinations':Daily_Vaccinations, 'Daily People Vaccinated':Daily_People_Vaccinated}
    features=pd.DataFrame(user_data,index=[0])
    result =""
    
    
    smt.write("## Your Chosen weightings: ")
    smt.write(features)
    
    smt.write("\n\n\n ### THE MODEL PREDICTS: ")
    
    prediction = round(model.predict(features)[0],0)
    smt.text(f"There will {prediction} deaths.")     
     
    
if __name__=='__main__':
    main()

 


 
    
