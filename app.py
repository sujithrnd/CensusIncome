from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,Predictionpipeline


application=Flask(__name__)

app=application


@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        val=request.form.get('hours-per-week')
        print('#####@@@@###hours-per-week::::',val)
        data=CustomData(
            age=float(request.form.get('age')),
            workclass = request.form.get('workclass'), 
            fnlwgt = int(request.form.get('fnlwgt')),
            education = request.form.get('education'),
            education_num = int(request.form.get('education_num')),
            marital_status = request.form.get('marital_status'),
            occupation = request.form.get('occupation'),
            relationship = request.form.get('relationship'),
            race= request.form.get('race'),
            sex = request.form.get('sex'),
            capital_gain =float(request.form.get('capital_gain')),
            capital_loss =float(request.form.get('capital_loss')),
            hours_per_week =float(request.form.get('hours-per-week')),
            native_country =request.form.get('native_country'),

        )
        
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=Predictionpipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('result.html',final_result=results)






if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)

