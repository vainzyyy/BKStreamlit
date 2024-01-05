import streamlit as st
import numpy as np
import pandas as pd
import re
import itertools

from google.colab import drive
drive.mount('/content/drive')

!ls "/content/drive/My Drive/Dataset/hungarian.data"

dir = '/content/drive/My Drive/Dataset/hungarian.data'

with open(dir, encoding='Latin1') as file:
  lines =[line.strip() for line in file]

lines[0:10]

data = itertools.takewhile(
  lambda x: len(x) == 76,
  (' '.join(lines[i:(i + 10)]).split() for i in range(0, len(lines), 10))
)
df = pd.DataFrame.from_records(data)
df.head()
