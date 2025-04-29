from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import pandas as pd
data=pd.DataFrame([
    {"Free":"Yes","Win":"Yes","Offer":"Yes","Spam":"Yes"},
    {"Free":"No","Win":"No","Offer":"Yes","Spam":"Yes"},
         {"Free":"No","Win":"No","Offer":"Yes","Spam":"Yes"},
            {"Free":"No","Win":"No","Offer":"No","Spam":"Yes"},
               {"Free":"No","Win":"No","Offer":"Yes","Spam":"No"},
               {"Free":"No","Win":"No","Offer":"Yes","Spam":"Yes"},
                  {"Free":"No","Win":"Yes","Offer":"No","Spam":"No"},
])

model=DiscreteBayesianNetwork([
    ("Spam","Win"),
    ("Spam","Offer"),
    ("Spam","Free")
])

model.fit(data,estimator=MaximumLikelihoodEstimator)
infer=VariableElimination(model)
res1=infer.query(variables=["Spam"],evidence={"Free":"Yes"})
res2=infer.query(variables=["Spam"],evidence={"Win":"Yes"})
res3=infer.query(variables=["Spam"],evidence={"Offer":"Yes"})
res4=infer.query(variables=["Spam"],evidence={"Free":"No","Offer":"No","Win":"No"})
print("Probability of ")
print(res1)
print("Probability of ")
print(res2)
print("Probability of ")
print(res3)
print("Probability of ")
print(res4)