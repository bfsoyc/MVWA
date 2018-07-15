# MVWA
This model caputures the semantic similarity between two short text(sentence). It's simple but very effective on some NLP task.<br>
We evaluate our model on 3 dataset: `SICK dataset`, `WikiQA dataset` and a authorized use only `LibBA dataset` derived from the abstract of publication. To run the model on sepcified dataset you have to prepare the data and the pre-train word vector at first.<br>
The code is writen in python2. If you are using python3, try:
		2to3 -w ./

-------
#### Take an example: Run the model on SICK dataset
Download dataset from: http://clic.cimec.unitn.it/composes/sick.html<br>
Unzip and copy the corresponding resource to ./data/SICK/SICK.txt
>We recommend you put the download file under ./data and unzip right there with the defautl name, as well as other resource.
 
Download the global word vector(Glove) from: https://nlp.stanford.edu/projects/glove/<br>
>The word vector we use in our experiment is the 300 dimension word vector from glove.6B.zip.
Unzip and Copy the corresponding resource to ./data/glove.6B/glove.6B.300d.txt

Run the `dataTransformSICK.py` python script in root dir.

Run the main script to train and evaluate the model with given configuration text file, in terminal:
		python main.py -config configSICK.txt

Enjoy your journey to metaphysics~

-------
This repository has been polished but still need improvement.<br>
Any suggestion and contribution is welcome.
