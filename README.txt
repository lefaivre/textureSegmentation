-----------------------------------------
Adam Lefaivre
Cpsc 5990
Final Program Project
Dr. Howard Cheng
-----------------------------------------

Ensure the programs are located in same place as the "testImages" folder

i.e.

whateverParentDir/gabor.py
whateverParentDir/moments.py
whateverParentDir/_utils.py
whateverParentDir/testImages

change directories to whateverParentDir


-----------------------------------------
Ensure these packages are installed:
-----------------------------------------
- cv2
- math
- numpy
- scipy (for signal, cluster, and ndimage)
- argparse
- sklearn (for cluster)
- matplotlib (for pyplot)
- glob
- os


-----------------------------------------
Running the Programs:
-----------------------------------------
1. Open a command line window

2. Change directories to where you stored the files: 
	gabor.py, moments.py, and _utils.py

3. Type either: 
	python gabor.py -h
	or,
	python moments.py -h
	
	This will bring up the help menu, so that you can
	see the variety of parameters that can be passed in.
	You will also be able to see which parameters are set to 
	default values.  If a default is not specified then the
	parameter is required by the program. 
	
4. To run one of the programs, on the command line type:
	python gabor.py -infile whatever/directory/img.png -outfile whatever/directory/imgOut.png -option1 value1 -option2 value2

5. Refer to the parameters section at the bottom of this README 
  to see the different parameters for each test image. 
  Feel free to copy/paste/modify any of these commands at your convenience.


-----------------------------------------
Optional:
-----------------------------------------
If you are wanting to call the functions to crop textures together, 
then please ensure these packages are installed:
- PIL (for Image, ImageOps, and ImageDraw)

The "Original" Brodatz textures were found here:
- http://multibandtexture.recherche.usherbrooke.ca/original_brodatz.html
- Click the download button at the bottom of the page 


-----------------------------------------
Parameters:
-----------------------------------------
GABOR FILTER RESULTS:

python gabor.py -infile ./testImages/G_Pair0.png -outfile ./out.png -k 2 -gk 17 -M 31 -sigma 7
python gabor.py -infile ./testImages/G_Nat5.png -outfile ./out.png -k 5 -gk 17 -M 31 -sigma 7
python gabor.py -infile ./testImages/G_Nat5.png -outfile ./out.png -k 5 -gk 17 -M 31 -sigma 7 -spw 2
python gabor.py -infile ./testImages/G_Nat16.png -outfile ./out.png -k 16 -gk 17 -M 35 -sigma 7 -spw 2
python gabor.py -infile ./testImages/G_HigherOrder0.png -outfile ./out.png -k 2 -gk 7 -M 41 -sigma 5
python gabor.py -infile ./testImages/G_HigherOrder1.png -outfile ./out.png -k 2 -gk 17 -M 49 -sigma 7
python gabor.py -infile ./testImages/G_HigherOrder2.png -outfile ./out.png -k 2 -gk 17 -M 49 -sigma 9 -gamma 0.5
python gabor.py -infile ./testImages/G_HigherOrder3.png -outfile ./out.png -k 2 -gk 13 -M 7 -sigma 7 


MOMENT-BASED RESULTS:

python moments.py -infile ./testImages/M_Pair0.png -outfile ./out.png -k 2 -W 9 -L 37 -i True -spw 0
python moments.py -infile ./testImages/M_Pair0.png -outfile ./out.png -k 2 -W 9 -L 37 -spw 1
python moments.py -infile ./testImages/M_Pair0.png -outfile ./out.png -k 2 -W 9 -L 37 -spw 2
python moments.py -infile ./testImages/M_Pair0.png -outfile ./out.png -k 2 -W 9 -L 37 -spw 3
python moments.py -infile ./testImages/M_Pair1.png -outfile ./out.png -k 2 -W 9 -L 49 
python moments.py -infile ./testImages/M_Pair2.png -outfile ./out.png -k 2 -W 9 -L 49 
python moments.py -infile ./testImages/M_Pair3.png -outfile ./out.png -k 2 -W 9 -L 49 
python moments.py -infile ./testImages/M_Nat4.png -outfile ./out.png -k 4 -W 9 -L 15
python moments.py -infile ./testImages/M_Nat4.png -outfile ./out.png -k 4 -W 9 -L 15 -pq 3
