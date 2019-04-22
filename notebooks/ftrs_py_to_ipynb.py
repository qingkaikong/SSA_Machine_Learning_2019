#Convert feature engg example python file to jupyter .ipynb 
#https://stackoverflow.com/questions/23292242/converting-to-not-from-ipython-notebook-format

from IPython.nbformat import v3, v4

with open("feature_engg_example.py") as fpin: 
    text = fpin.read() 

nbook = v3.reads_py(text)
nbook = v4.upgrade(nbook)  # Upgrade v3 to v4

jsonform = v4.writes(nbook) + "\n"
with open("03_feature_engg_examples.ipynb", "w") as fpout:
    fpout.write(jsonform)

