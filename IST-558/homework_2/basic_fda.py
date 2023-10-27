import skfda
from skfda.exploratory.visualization import FPCAPlot
from skfda.preprocessing.dim_reduction import FPCA

import plotly.express as px
import pandas as pd

data = pd.read_csv("./data/hw2p3_data.csv")
# data.head()

fpca_discretized = FPCA(n_components=7)
fpca_discretized.fit(data)
fpca_discretized.components_.plot()