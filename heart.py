import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
from st_aggrid import AgGrid
st.set_page_config(page_title="Heart Disease Prediction", page_icon="🫀",layout='wide')
from streamlit_option_menu import option_menu
selected = option_menu(
        menu_title = None,
        options = ["Home", "Medical Status", "Lifestyle Status","Social Status","Model EDA","Model App"],
        icons=['house', 'bi bi-person-workspace', "bi bi-person-bounding-box", 'bi bi-person-lines-fill','gear'],
        default_index=0,
        orientation ='horizontal'
        )

st.expander('Expander')
with st.sidebar:
    uploaded_file = st.file_uploader("Choose a file")
    df = pd.read_csv(uploaded_file, encoding = 'ISO-8859-1')    
if selected == "Home":
    st.title("HomePage")
    from PIL import Image
    image = Image.open('/Users/kassemhajjoul/Desktop/Healthcare Analytics/heartdisease.png')
    st.image(image,width=500)
    st.write("In this tool, the dashboard will present the medical, social and lifestyle status of the patient to analyze the correlation between these factors and the risks of a heart-disease.")
    st.write("The tool will futhermore predict the user test case to assess the model.")
    AgGrid(df)
    st.header("Data table:")
    st.markdown("This dataset contains 253,680 survey responses from cleaned BRFSS 2015 to be used primarily for the binary classification of heart disease. ")
    df.info()
if selected == "Medical Status":
    kp1,kp2,kp3,kp4=st.columns(4)
    kp1.metric(label="Total study cases", value= "256k")
    kp2.metric(label="Total Smokers", value= "112k")
    kp3.metric(label="Total patients with healthcare", value= "241k")
    kp4.metric(label="Patients with Diabetes", value="40k")
    col1,col2=st.columns([0.5,0.5])

    with col1:
        df_1 = df.groupby(["HighBP"])["ID"].count().reset_index(name="Count")
        p1=px.bar(df_1, x="HighBP",y='Count',width=450, height=390,title='High Blood-Pressure Patients')
                         #   "Year_of_response": "Year",
                          #  "sum": "Annual Premium",
                           # },title="Annual Premium by Year")
        st.plotly_chart(p1)
        df_2 = df.groupby(["Diabetes"])["ID"].count().reset_index(name="Count")
        p2=px.bar(df_2, x="Diabetes", y="Count",width=450, height=390, title='Diabetes Patients')
        st.plotly_chart(p2)

        with col2:
            df_3= df.groupby(["Stroke"])["ID"].count().reset_index(name="counts")
            p3=px.sunburst(df_3,path = ["Stroke"],values = "counts",color = "counts", width=400,height = 400,
            title="Records of Previous Strokes")
            st.plotly_chart(p3)
            df_4= df.groupby(["HighChol"])["ID"].count().reset_index(name="counts")
            p4=px.pie(df_4, values='counts', names='HighChol', title='High Cholesterol Patients',width=400,height=380)
            st.plotly_chart(p4)
if selected=="Lifestyle Status":
            col3,col4=st.columns([0.5,0.5])
            with col3:
                df_5= df.groupby(["Smoker"])["ID"].count().reset_index(name="Count")
                p5=px.bar(df_5,x='Smoker', y="Count", title= "Smoking Ratio of Patients")   
                st.plotly_chart(p5)
                df_6= df.groupby(["HvyAlcoholConsump"])["ID"].count().reset_index(name="Count")
                p6=px.bar(df_6,x='HvyAlcoholConsump', y="Count", title= "Heavy Alcohol Consumption Ratio of Patients")  
                st.plotly_chart(p6)
            with col4:
                df_7 = df.groupby(["DiffWalk"])["ID"].count().reset_index(name="Count")
                p7=px.pie(df_7, values='Count', names='DiffWalk', title='Walking Exercise',width=450, height=430)
                st.plotly_chart(p7)
                df_8= df.groupby(["Fruits"])["ID"].count().reset_index(name="Count")
                p8=px.pie(df_8, values='Count', names='Fruits', title='Fruits Consumption',width=450, height=430)
                st.plotly_chart(p8)
if selected=="Social Status":
            col5,col6=st.columns([0.5,0.5])
            with col5:
                df_9 = df.groupby(["Education"])["ID"].count().reset_index(name="Count")
                p9=px.treemap(df_9, path=["Education"], values='Count',width=400,height=420,title="Distribution of Patients By Income Levels")
                st.plotly_chart(p9)
                df_10 = df.groupby(["Income"])["ID"].count().reset_index(name="Count")
                p10=px.bar(df_10, x="Income",y="Count", title='Distribution of Patients by Education Levels',width=400,height=420)
                st.plotly_chart(p10)
            with col6:
                df_11 = df.groupby(["AnyHealthcare"])["ID"].count().reset_index(name="Count")
                p11=px.pie(df_11, values='Count', names='AnyHealthcare', title='Distribution of Patients by Healthcare Availability',width=400,height=420)
                st.plotly_chart(p11)
                df_12 = df.groupby(["NoDocbcCost"])["ID"].count().reset_index(name="Count")
                p12=px.pie(df_12, values='Count', names='NoDocbcCost', title='Patients who do not Visit the Doctor Due to Costs',width=400,height=420)
                st.plotly_chart(p12)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis 
from sklearn.decomposition import PCA
if selected=="Model EDA": 
            col7,col8=st.columns([0.5,0.5])
            with col7:
                df_13 = df.groupby(["HeartDiseaseorAttack"])["ID"].count().reset_index(name="Count")
                p13=px.bar(df_13, x="HeartDiseaseorAttack", y="Count",width=500, height=520, title='Target Label')
                st.plotly_chart(p13)
            with col8:
                corr_matrix = df.corr()
                p14=sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
                st.pyplot(p14)

if selected=="Model App": 
            y = df.HeartDiseaseorAttack
            x = df.drop(["HeartDiseaseorAttack","ID"],axis = 1)
            columns = x.columns.tolist()
            X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            X_train_df = pd.DataFrame(X_train, columns = columns)
            X_train_df_describe = X_train_df.describe()
            X_train_df["target"] = Y_train
            knn = KNeighborsClassifier(n_neighbors = 2)
            knn.fit(X_train, Y_train)
            y_pred = knn.predict(X_test)
            cm = confusion_matrix(Y_test, y_pred)
            acc = accuracy_score(Y_test, y_pred)
            st.write("CM: ",cm)
            st.write("Basic KNN Acc: ",acc)
            #Side bar inputs for prediction
            # User Data Entry
            def user_report():
                Diabetes = st.slider('Diabetes',min_value =0,max_value=1,step=1)
                Stroke= st.slider('Previous Stroke',min_value =0,max_value=1,step=1)
                Smoker=st.slider('Smoker',min_value=0,max_value=1,step =1)
                CholCheck= st.slider('CholCheck',min_value=0,max_value=1,step =1)
                HighBP= st.slider('HighBP',min_value=0,max_value=1,step =1)
                Age= st.slider('Age',min_value=5,max_value=100,step =1)
                PhysActivity= st.slider('Physical Activity',min_value=0,max_value=1,step =1)
                Fruits= st.slider('Fruits',min_value=0,max_value=1,step =1)
                Veggies= st.slider('Veggies',min_value=0,max_value=1,step =1)
                AnyHealthcare=st.slider('Any Healthcare',min_value=0,max_value=1,step =1)
                HvyAlcoholConsump=st.slider('Heavy Alcohol Consumption',min_value=0,max_value=1,step =1)
                NoDocbcCost=st.slider('No Doctor Because Cost',min_value=0,max_value=1,step =1)
                GenHlth=st.slider('No Doctor Because Cost',min_value=1,max_value=5,step =1)
                Education=st.slider('Education Level',min_value=1,max_value=6,step =1)
                Income=st.slider('Income Level',min_value=1,max_value=8,step =1)
                Sex= st.slider('Sex',min_value=0,max_value=1,step =1)
                MentHlth=st.slider('Mental Health',min_value=1,max_value=30,step =1)
                PhysHlth=st.slider('Physical Health',min_value=1,max_value=30,step =1)
                DiffWalk=st.slider('Walk',min_value=0,max_value=1,step =1)
                BMI=st.slider('BMI',min_value=12,max_value=98,step =1)
                HighChol=st.slider('High Cholesterol',min_value=0,max_value=1,step =1)
                user_report_data = {
                'Diabetes': Diabetes,
                'HighBP': HighBP,
                'HighChol': HighChol,
                'CholCheck': CholCheck,
                'BMI': BMI,
                'Smoker': Smoker,
                'Stroke' : Stroke,
                'Fruits': Fruits,
                'Veggies': Veggies,
                'PhysActivity': PhysActivity,
                'HvyAlcoholConsump': HvyAlcoholConsump,
                'AnyHealthcare': AnyHealthcare,
                'NoDocbcCost': NoDocbcCost,
                'GenHlth': GenHlth,
                'PhysHlth': PhysHlth,
                'DiffWalk': DiffWalk,
                'MentHlth':MentHlth,
                'Sex': Sex,
                'Age': Age,
                'Education': Education,
                'Income': Income,
                }
                report_data =pd.DataFrame(user_report_data, index = [0])
                return report_data
            user_data = user_report()
            
    
            st.header('Guest Data')
            st.table(user_data)
            testpred=knn.predict(user_data)
            st.write(testpred)

    

  
