from plotly.subplots import make_subplots
import plotly.graph_objects as go
from skimage import io
img = io.imread('https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Crab_Nebula.jpg/240px-Crab_Nebula.jpg')
fig = make_subplots(
    rows=1, cols=2)
fig.add_trace(go.Image(z=img), 1, 1)
fig.add_trace(go.Image(z=img + 1), 1, 1)
fig.show()

print('donw')