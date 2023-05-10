import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object 
import pandas as pd


class Predictionpipeline:
    def __ini__(self):
        pass
    def predict(self,features):
        try:
            logging.info('start predict ')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            logging.info('Load preprocessor.pkl  with preprocessor_path=',preprocessor_path)
            preprocessor=load_object(preprocessor_path)
            logging.info('Load model.pkl  with model_path=',model_path)
            model=load_object(model_path)
            logging.info('transform feautures')
            data_scaled=preprocessor.transform(features)
            logging.info('Predict')
            pred=model.predict(data_scaled)
            logging.info('Return prediction ')
            return pred
        except Exception as e:
            logging.info('Exception occured in Prediction')
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,                 
                age            :float, 
                workclass      :str,
                fnlwgt         :float,
                education      :str,
                education_num  :float ,
                marital_status :str,
                occupation     :str,
                relationship   :str,
                race           :str,
                sex            :str,
                capital_gain   :float ,
                capital_loss   :float ,
                hours_per_week :float ,
                native_country :str):
                #Class          :str):
                
                self.age            = age            
                self.workclass      = workclass      
                self.fnlwgt         = fnlwgt         
                self.education      = education      
                self.education_num  = education_num  
                self.marital_status = marital_status 
                self.occupation     = occupation     
                self.relationship   = relationship   
                self.race           = race           
                self.sex            = sex            
                self.capital_gain   = capital_gain   
                self.capital_loss   = capital_loss   
                self.hours_per_week = hours_per_week 
                self.native_country = native_country 
                #self.Class        = Class

    

    def  get_data_as_dataframe(self):
        try:

            custom_data_input_dict = {
                'age':[self.age],
                'workclass':[self.workclass],
                'fnlwgt':[self.fnlwgt],
                'education':[self.education],
                'education_num':[self.education_num],
                'marital_status':[self.marital_status],
                'occupation':[self.occupation],
                'relationship':[self.relationship],
				'race':[self.race],
				'sex':[self.sex],
				'capital_gain':[self.capital_gain],
				'capital_loss':[self.capital_loss],
				'hours_per_week':[self.hours_per_week],
				'native_country':[self.native_country]#,
					# 'Class':[self.Class]
            }	
            df=pd.DataFrame(custom_data_input_dict)
            logging.info('Data Frame gathered')
            return df
        except Exception as e:
            logging.info('Exception occured prediction pipeline')
            raise CustomException(e,sys)
            
            
            
            
            



