#%%

import pandas as pd

df = pd.read_excel(r"C:\Users\filip\Desktop\codigos\Projetos - Teo My Whay\TEOMEWHY - CIÊNCIA DE DADOS\Machine Learn - 2025\machine-learning-2025\data\dados_cerveja.xlsx")

df.head()
# %%

# Ferature são ás Caracrterísticas variveis;
features = ["temperatura","copo","espuma","cor"]

# Target seria a variável resposta;
target = "classe"

X = df[features]
y = df[target]

X = X.replace({"mud":1,"pint":2,"sim":1,"não":0, "clara":0, "escura":1})

# %%

from sklearn import tree

model = tree.DecisionTreeClassifier()

model.fit(X=X, y=y)
# %%
import matplotlib. pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(model, feature_names=features, class_names=model.classes_, filled=True)

# %%
