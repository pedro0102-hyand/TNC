import joblib
import numpy as np

modelo = joblib.load("models/baseline.joblib")
termos = np.array(modelo.named_steps["tfidf"].get_feature_names_out())
coef = modelo.named_steps["clf"].coef_
print("Formato de coef_:", coef.shape)

for i, classe in enumerate(modelo.classes_):
    top = termos[np.argsort(coef[i])[-10:][::-1]]
    print(f"{classe:<15} {', '.join(top)}")