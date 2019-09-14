import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():


    # Random Forest Regression

    # Importing the libraries 
    import pandas as pd

    # Importing the dataset
    dataset = pd.read_csv('commodity1.csv')
    X = dataset.iloc[:, 0:4].values
    y = dataset.iloc[:, 4:5].values

    # Encoding categorical data(region)
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder1 = LabelEncoder()
    X[:, 3] = labelencoder1.fit_transform(X[:, 3])
    onehotencoder1 = OneHotEncoder(categorical_features = [3])
    X = onehotencoder1.fit_transform(X).toarray()

    # Encoding categorical data(commodity)
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder2 = LabelEncoder()
    X[:, 1] = labelencoder2.fit_transform(X[:, 1])
    onehotencoder2 = OneHotEncoder(categorical_features = [1])
    X = onehotencoder2.fit_transform(X).toarray()


    # Encoding categorical data(center)
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder3 = LabelEncoder()
    X[:, 2] = labelencoder3.fit_transform(X[:, 2])
    onehotencoder3 = OneHotEncoder(categorical_features = [2])
    X = onehotencoder3.fit_transform(X).toarray()

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y)

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Fitting Random Forest Regression to the dataset
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(X_train, y_train)

    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    #1
    final_features[1] = labelencoder1.fit_transform(final_features[1])
    final_features = onehotencoder1.fit_transform(final_features).toarray()
     #1
    final_features[2] = labelencoder2.fit_transform(final_features[2])
    final_features = onehotencoder2.fit_transform(final_features).toarray()
     #1
    final_features[3] = labelencoder3.fit_transform(final_features[3])
    final_features = onehotencoder3.fit_transform(final_features).toarray()

    prediction = regressor.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Expected Commodity Price should be $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)