# USAGE: python facepointChooser.py -i <image file0> [...] [-l <logfile>]
# Left mouse click: selects point
# Right mouse click: undo select
# Image window close: writes selected points to file
# Root window click 'Quit': closes program

import Tkinter as tk
from PIL import Image, ImageTk
import argparse, sys

parser = argparse.ArgumentParser(description="Get corresponding coordinates by clicking on the image")
parser.add_argument("-l", "--logFile", help="Name of output file")
parser.add_argument("-i", "--image", help="one face image", nargs="+", required=True)
args = parser.parse_args()

print "Output file:%s"%args.logFile
print "********************"

if not args.logFile:
    args.logFile = "coordinates.txt"

images = []
for i in args.image:
    images.append(i)

# file to write all coordinates to
coordLogFile = open(args.logFile, "w")

# create an invisible global main root for all windows
root = tk.Tk()
# Root must be closed last, after writing all chosen coordinates in
#  each image to the file, by clicking close on each image window
#  else they will not be written
message = "Click close button on image windows to save coordinates,\nTHEN me to end"
label = tk.Label(root, text=message)
label.pack()
button = tk.Button(root, text="Quit", command=root.destroy)
button.pack()

class GWindow(tk.Canvas):
    def __init__(self, img, width=500, height=50):
        master = tk.Toplevel(root)
        master.protocol("WM_DELETE_WINDOW", self.close)
        tk.Canvas.__init__(self, master, width=width, height=height)
        master.resizable(0,0) # no resizing
        master.title(img) # window title label <--- self.master.title (?)
        
        self.bind("<Button-1>", self.onLClick) #store coord
        self.bind("<Button-2>", self.onRClick) #undo coord-storage
        self.pack() # fit given image?

        # allow opening of more than just .png/.ppm images
        self.img = Image.open(img)
        self.img = self.img.resize((width, height))
        self.photo = ImageTk.PhotoImage(self.img)
        self.create_image(0, 0, anchor="nw", image=self.photo)
        self.name = img
        # add 4 corners of the image by default
        imR, imC = self.img.size;
        imR -= 1
        imC -= 1
        self.clicks = [(0, 0), (imR, 0), (imR, imC), (0, imC)]
        self.polygons = [] # drawn points where clicked
        
        self.writtenOnClose = False

    # 'on left mouse button click' callback
    # Store user's choice of significant coordinates
    def onLClick(self, event):
        x, y = event.x, event.y
        # draw RED point where coordinate is selected
        self.polygons.append(self.create_polygon(x-2, y-2, x+2, y-2,x, y+2, fill='red'))
        # to account for default padding
        x -= 4
        y -= 5
        print "%s: %s, %s" %(self.name, x, y)
        self.clicks.append((x, y)) # tuple (x, y)

    # 'on right mouse button click' callback
    # Undo store of last coordinates
    def onRClick(self, event):
        try:
            x, y = self.clicks.pop()
            lastPolygon = self.polygons.pop()
            self.delete(lastPolygon)
            print "Undid last point", x, y
        except (IndexError):
            print "! No more coordinates stored"

    # close handler (when window closed)
    def close(self):
        self.img.save("pts_"+self.name[:-4]+".png", "png")
        if self.writtenOnClose:
            print "*%s Closed already" % self.name
        else:
            print "***%s good close" % self.name
            coordLogFile.write("***%s\n" % self.name)
            for c in self.clicks:
                print "x:%s\t y:%s" %(c[0], c[1])
                coordLogFile.write("%s %s\n"%(c[0],c[1]))
        # don't repeat:
        self.writtenOnClose = True 


# Open each image in its own window
for i in args.image:
    #i = args.image
    w, h = Image.open(i).size
    win = GWindow(i, width=int(w)*2, height=int(h)*2)

# create windows when program runs
root.mainloop()

coordLogFile.close()
