*****************************
Usage
*****************************
How to use facepointChooser.py, faceMorpher, and stochopt.py (simulated annealing)
Example commands
Files (results/ testing)

** facepointChooser.py

$ python facepointChooser.py -l <log file name> -i <image0> [<image1> …]

All chosen coordinates will be logged in one log file. Simply copy and paste each relevant section (headed by “ **<namei>”) into separate log files, and put the files into the same directory as the faceMorpher build. Sample log files are included for reference/ testing use.

** faceMorpher

For basic help information, just run $ ./faceMorpher for help information.
Note: options can be input in any order.
The executable was built in Mac 10.7+, OpenCV 2.4.5.
Put executable in the same directory as images/ landmarks files (if morphing), and same directory as folders (if getting mean/ eigenfaces)

Possible compilation issues (XCode settings):
. Header Search Paths: /usr/local/include
. Library Search Paths: /usr/local/lib
. C++ Standard Libary: libstdc++ (GNU C++ standard library)

******************************
Prepared files for results:
***
Morphing (goto test0 dir): 
. 01happy.jpg, 01ref.txt, 02centerlight.jpg, 02ref.txt
. results in test0_movie
***
Used to make rmMorphFast.mp4 (goto rm dir):
. Images and landmark files in rm directory.
. results in rm_movie
***
Mean face/ Eigenfaces (goto mean_eigen dir):
. images in cleanMeanData directory, landmarks in meanFiles directory
. results generated can be organized with script "mrClean". $ bash mrClean
******************************

- Facemorphing
$ ./faceMorpher -i0 <image0> -f0 <image0’s landmarks> -i1 <image1> -f1 <image1’s landmarks>

EXAMPLE: ./faceMorpher -i0 01happy.jpg -f0 01ref.txt -i1 02centerlight.jpg -f1 02ref.txt

- Mean face
$ ./faceMorpher -m ./<images to make mean>/ -mf ./<faces-for-mean’s landmarks directory>/ 

EXAMPLE: ./faceMorpher -m ./cleanMeanData/ -mf ./meanFiles/ -e 1

- Eigenfaces
$ ./faceMorpher -m ./<images to make mean>/ -mf ./<faces-for-mean’s landmarks directory>/  -e <1|0>

-e 1: eigenface with normalized shape (for AAMs)
-e 0: eigenface directly from images (normal)

**Simulated annealing

$ python stochopt.py

Dependencies: Python 2.7+, Tkinter
