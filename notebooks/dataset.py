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
import matplotlib.pyplot as plt
import pandas as pd

# %%
# figsize = (12, 8)
figsize = (16, 9)

# %%
data_directory = pathlib.Path("../data")

# %%
# dataset_filename = "1_pct_dataset.pickle"
dataset_filename = "5_pct_dataset.pickle"
# dataset_filename = "20_pct_dataset.pickle"
# dataset_filename = "dataset.pickle"

dataset_path = data_directory / dataset_filename
dataset = pd.read_pickle(dataset_path)

# %%
dataset.head()

# %%
dataset.sample(10, random_state=7).sort_index()

# %%
dataset.info()

# %%

# %%

# %%

# %% [markdown] tags=[]
# ## stats

# %%
dataset["transcript_id"].value_counts()

# %%
dataset["sequence"].value_counts().values

# %%

# %%

# %%

# %% [markdown] tags=[]
# ## sequence length

# %%
dataset["sequence_length"] = dataset["sequence"].str.len()

# %%
dataset.head()

# %%

# %%
dataset["sequence_length"].sort_values()

# %%

# %%
dataset.loc[dataset["sequence_length"].sort_values().index][-10:]

# %%

# %%
figure = plt.figure()
ax = dataset["sequence_length"].hist(figsize=figsize, bins=512)
ax.axvline(
    x=round(dataset["sequence_length"].mean() + 0.5 * dataset["sequence_length"].std()),
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
