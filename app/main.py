from embedding import embding_text
import numpy as np
import joblib
from dotenv import load_dotenv
import os

load_dotenv()

path_model= os.getenv("model_path")
query = "Customer Support Inquiry,Seeking information on digital strategies that can aid in brand growth and details on the available services. Looking forward to learning more to help our business grow. Thank you, and I look forward to hearing from you soon."

embding_query = embding_text(query)

embding_query = np.array(embding_query).reshape(1, -1)
def get_function_TEST(embding_query):
    v = path_model
    if (!path_model):
        return none 
    else:
        return load_dotenv(path_model)

mdoel = joblib.load(path_model)


prediction = mdoel.predict(embding_query)[0]

if prediction == 0:
    prediction = "Incident"
elif prediction == 1:
    prediction = "Request"
elif prediction == 2:
    prediction = "Problem"
else :
    prediction = "Change"
print(prediction) 
