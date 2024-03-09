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
html_temp = """
    <div style ="background-color:blue ;padding:13px">
    <h1 style ="color:black;text-align:center;">Covid-19 Deaths Prediction ML Model</h1>
    </div>
    """
#smt.title("Covid-19 deaths Prediction Model")
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed

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
smt.header("DATA PREPARATION")
#smt.write("Below are steps that were taken to clean and prepare the data:")
smt.write("Data Shape")
smt.write(df.shape)
smt.write("Total number of null values in each column:")
smt.write(df.isnull().sum())
smt.write("We will then drop all null values in each column")
smt.write(df.dropna(inplace=True))
smt.write(df.isnull().sum())
smt.title("ANALYSIS")
smt.header("Correlations between different variables")
fig=plt.figure()
sns.heatmap(df[['Weekly Cases', 'Weekly Deaths', 'Total Vaccinations', 'People Vaccinated',
       'People Fully Vaccinated', 'Total Boosters', 'Daily Vaccinations', 'Daily People Vaccinated', "Next Week's Deaths"]].corr(), cmap='Blues', annot=True)
#sns.heatmap(heatmap_data, annot=True)
smt.pyplot(fig)
smt.write("From the above we can some Regressor variables such as 'Total Boosters' contribute too little to the model therefore we shall drop such columns")
smt.write("The following is data after removing some attributes")
df.drop("Total Boosters",axis=1,inplace=True)
df.drop("People Fully Vaccinated",axis=1,inplace=True)
df.drop("People Vaccinated",axis=1,inplace=True)
df.drop("Total Vaccinations",axis=1,inplace=True)
df.drop("Weekly Cases",axis=1,inplace=True)
df.drop("Daily People Vaccinated",axis=1,inplace=True)
x=df[['Weekly Deaths','Daily Vaccinations']].fillna("0")
y=df["Next Week's Deaths"]
smt.write(x,y)
smt.write("For best regression model, the regressor variables with a higher correlation should be eliminated/removed. There this heatmap displays the two variables that contribute much to the model and have less regression to one another:")
fig2=plt.figure()
sns.heatmap(df[['Weekly Deaths', 'Daily Vaccinations', "Next Week's Deaths"]].corr(), cmap='Blues', annot=True)
smt.pyplot(fig2)
x=df["Weekly Cases"]
y=df["Next Week's Deaths"]
plt.scatter(x,y,color="blue")
#smt.pyplot(fig)
#smt.write(sns.displot(x="Weekly Cases",kde=True,data=df,bins=5))

#smt.write(sns.scatterplot(x="Weekly Cases",y="Next Week's Deaths",data=df,hue="Total Vaccinations"))
#smt.header("Altair chart")
#smt.altair_chart(df)

    
def prediction(Weekly_Cases,Weekly_Deaths,Daily_Vaccinations,Daily_People_Vaccinated):  
   
    prediction=model.predict([[Weekly_Cases,Weekly_Deaths,Daily_Vaccinations,Daily_People_Vaccinated]])
    print(prediction)
    return prediction
def main():
    smt.sidebar.header("Choose your inputs")
    #Weekly_Cases=smt.sidebar.number_input("Weekly_Cases")
    Weekly_Deaths=smt.sidebar.number_input("Weekly_Deaths")
    #Total_Vaccinations=smt.sidebar.number_input("Total_Vaccinations")
    #People_Vaccinated=smt.sidebar.number_input("People_Vaccinated")
    #People_Fully_Vaccinated=smt.sidebar.number_input("People_Fully_Vaccinated")
    #Total_Boosters=smt.sidebar.number_input("Total_Boosters")
    Daily_Vaccinations=smt.sidebar.number_input("Daily_Vaccinations")
    #Daily_People_Vaccinated =smt.sidebar.number_input("Daily_People_Vaccinated")
    
    
    user_data={'Weekly Deaths':Weekly_Deaths, 'Daily Vaccinations':Daily_Vaccinations}
    features=pd.DataFrame(user_data,index=[0])
    result =""
    
    
    smt.write("## Your Chosen weightings: ")
    smt.write(features)
    
    smt.write("\n\n\n ### THE MODEL PREDICTS: ")
    
    prediction = round(model.predict(features)[0],0)
    smt.text(f"There will {prediction} deaths.")     
     
    
if __name__=='__main__':
    main()

 


 
    
