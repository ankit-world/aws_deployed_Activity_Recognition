# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            Time=float(request.form['Time'])
            frontal = float(request.form['Acceleration for frontal axis'])
            vertical  = float(request.form['Acceleration for vertical axis'])
            lateral  = float(request.form['Acceleration for lateral axis'])
            Id = int(request.form['Id'])
            RSSI = float(request.form['RSSI'])
            Phase = float(request.form['Phase'])
            Frequency = float(request.form['Frequency'])

            data=pd.read_csv('new_file.csv')
            sc = StandardScaler()
            sc.fit_transform(data)
            x=sc.transform([[vertical,lateral,Id,RSSI,Phase,Frequency]])



            filename = 'bagg_dt.pickle'
            loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage
            # predictions using the loaded model file
            prediction=loaded_model.predict(x)
            print('prediction is', prediction[0])
            # showing the prediction results in a UI
            if prediction[0]==1:
                return render_template('predict.html',prediction='sit on bed')
            elif prediction[0]==2:
                return render_template('predict.html',prediction='sit on chair')
            elif prediction[0]==3:
                return render_template('predict.html',prediction='lying')
            else:
                return render_template('predict.html', prediction='ambulating')

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app