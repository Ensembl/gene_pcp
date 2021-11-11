# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pathlib

# %%
import pandas as pd

# %%
#pd.set_option("display.max_rows", 100)
#pd.set_option("display.max_columns", 100)
#pd.set_option("display.width", 1000)

# %%
import matplotlib.pyplot as plt

# %%
# figsize = (12, 8)
figsize = (16, 9)

# %%
data_directory = pathlib.Path("../data")

# %%
dataset_path = data_directory / "dataset.pickle"
data = pd.read_pickle(dataset_path)

# %%
data.head()

# %%
data.sample(10, random_state=5).sort_index()

# %%
data.info()

# %%

# %%

# %%

# %% [markdown] tags=[]
# ## stats

# %%
data["transcript_id"].value_counts()

# %%
data["sequence"].value_counts().values

# %%

# %%

# %%

# %% [markdown] tags=[]
# ## sequence length

# %%
data["sequence_length"] = data["sequence"].str.len()

# %%
data.head()

# %%

# %%
data["sequence_length"].sort_values()

# %%

# %%
data.iloc[data["sequence_length"].sort_values().index[-10:]]

# %%

# %%
figure = plt.figure()
ax = data["sequence_length"].hist(figsize=figsize, bins=512)
ax.axvline(
    x=round(data["sequence_length"].mean() + 0.5 * data["sequence_length"].std()),
    color="r",
    linewidth=1,
)
ax.set(xlabel="sequence length", ylabel="number of sequences")
figure.add_axes(ax)

# %%

# %%

# %%

# %%

# %%
