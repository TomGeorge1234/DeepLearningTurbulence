readme.txt



This here folder contains everything used in the project "Applications of Neural Networks to the Prediction of Tracer Fluxes in Quasi-Geostrophic Turbulence"




Author: Tom George
Data readme last updated: Oct 2018
Institution: Caltech
Email: tomgeorge1@btinternet.com




TO USE:
To use (i.e. train a neural network on QG data) simply open QGmain.py and follow the instructions on there before running. 




ITEMS:
•QGmain.py is the primary body of code. It loads the data then manipulates it in any desired way, then interfaces with the neural network architecture (saved in folder Networks) to train and display output. It  saves relevant data at the end. Each run is assinged a "savekey" which can be used to isolated results, saves networks etc.
•Networks contains the tensor flow code for various architectures used. Usually labelled something like NET3c2f.py where 3c = 3 convolutional layers, 2f = 2 fully connected layers. 
•getdata.py pull data from the data folders, loads it as numpy arrays, concatenates, slices, crops, normalises, finds the flux etc, applies filters if required then saves these back into the the folders 'fluxes' and 'fields' as numpy arrays which can be accessed my QGmain.py. If you need to use this there are two functions called save_field and save_flux which do it all, they have four highly logical inputs.
•figure_code.py is just used to plot various figures, there are other older figure codes in the other folder.
•updategit.sh is a shell script for updating the github repository. There is also a hidden file .gitignore which prevents certain folders from syncing.
•QG_matlab contains all the Matlab code used to simulate the baroclinic turbulence. 
•other contains various bits of code including and folder called old_code which has code use pre-15th Aug when we change the data from data32_1000 (now deleted) to data256_4000 along with how the data is manipulated.
•results contains a text file with a printed summary of any completed network training session
•Movies contains some movie files
•figures1&2 contains figures used for interim reports 1 and 2
•figures3 contains figures used after this


This code works best in an interactive python3 environment (we used Anaconda Spyder). 

NOTE: if the code uses LATEX text rendering (has something like: from matplotlib import rc, plt.rc('text',usetex=True) at the top) then this is almost always best run from the terminal window. Reset everything after use.

NOTE FOR GITHUB VIEWERS: The code QGmain won't actually work, since the data isn't saved onto GitHub as it is enormous. Please contact me and I can arrange to have the data sent to you. 




TO UPDATE GITHUB:
open terminal in current folder
>>> ./updategit.sh
then follow instructions
