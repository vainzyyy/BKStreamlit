# streamlit umumnya diinisialisasi dengan 'st'
import streamlit as st

st.set_page_config(
    page_title = "Estimasi CO2 Emissions",
    page_icon = ':car: | :red_car:' #nama emoji 
)

# st.write dapat digunakan menampilkan test,dataframe,visualisasi
st.title('Estimasi CO2 Emissions')
st.write('information about Co2 Emission')


# st.sidebar dapat digunakan untuk membuat sidebar
st.sidebar.header("User Input Features")

def user_input_features():
    EngineSize = st.sidebar.number_input('Insert a Engine Size', 1.0 , 8.4) #(label,minvalues,maxvalues,initial values)
    Cylinders = st.sidebar.number_input('Insert a Cylinders', 3 , 16)
    FuelConsumptionTotal = st.sidebar.number_input('Insert a Total Fuel Comsumption (L/100 km)', 1.0 , 25.8)
    
    data = {'ENGINESIZE': EngineSize,
            'CYLINDERS': Cylinders,
            'FUELCONSUMPTION_COMB': FuelConsumptionTotal
            }
    
    features = pd.DataFrame(data,index=[0])
    return features

df = user_input_features()
st.header("User Input Features")
st.write(df)

dfModel = pd.read_csv('data/FuelConsumption.csv')
dfModel = dfModel[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

df2 = dfModel.iloc[:,:-1]
if df.columns.equals(df2.columns):
    st.write("Column names are the same.")
else:
    st.write("Column names are different.")

# separate the feature and target
X = dfModel.iloc[:,:-1]
Y = dfModel.iloc[:,-1]

# splitting data into data train and test
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3 ,random_state = 42)

# creating model
LinearReg = LinearRegression()
LinearReg.fit(x_train,y_train)

st.write('press button below to predict : ')
if st.button('Predict'):
    # create progres bar widget with initial progress is 0%
    bar = st.progress(0)
        # create an empty container or space
    status_text = st.empty()
    for i in range(1,101):
        # create a text to showing a percentage process
        status_text.text("%i%% complete" %i)
        # give bar progress values
        bar.progress(i)
        # give bar progress time to execute the values
        time.sleep(0.01)

    ypred = LinearReg.predict(df)
    ypred = float(ypred[0])
    ypred = "{:.2f}".format(ypred)

    st.subheader('Prediction')
    st.metric('CO2 Emission',ypred,' g/km')

