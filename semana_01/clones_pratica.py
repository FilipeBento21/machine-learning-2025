#%%

import pandas as pd

df = pd.read_parquet(r"C:\Users\filip\Desktop\codigos\Projetos - Teo My Whay\TEOMEWHY - CIÊNCIA DE DADOS\Machine Learn - 2025\machine-learning-2025\data\dados_clones.parquet")

df.head()
df["General Jedi encarregado"].unique()


# %%
features = ["Massa(em kilos)","Estatura(cm)"]

# Target seria a variável resposta;
target = "Status "

X = df[features]
y = df[target]
# %%

from sklearn import tree

model = tree.DecisionTreeClassifier()

model.fit(X=X, y=y)
# %%
import matplotlib. pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(model, feature_names=features, class_names=model.classes_, filled=True, max_depth=3)
# %%
df.groupby("Status ").mean()
# %%
