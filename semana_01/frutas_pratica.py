
#%%
import pandas as pd

df = pd.read_excel(r"C:\Users\filip\Desktop\codigos\Projetos - Teo My Whay\TEOMEWHY - CIÊNCIA DE DADOS\Machine Learn - 2025\machine-learning-2025\data\dados_frutas.xlsx")
df

# %%
from sklearn import tree

arvore = tree.DecisionTreeClassifier(random_state=42)
# %%

respostas = df["Fruta"]

caracteristicas = ["Arredondada","Suculenta","Vermelha","Doce"]
carac_variaiveis = df[caracteristicas]

#%%
arvore.fit(carac_variaiveis, respostas)
# %%
arvore.predict([[0,0,0,0]])

# %%

import matplotlib.pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(arvore, feature_names=caracteristicas, class_names=arvore.classes_, filled=True)

# %%


proba = arvore.predict_proba([[1,1,1,1]])[0]

pd.Series(proba, index=arvore.classes_)
# %%
