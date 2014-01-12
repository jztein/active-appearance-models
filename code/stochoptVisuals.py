import math
import time

from Tkinter import *

class pointVisualization:
    def __init__(self, point1, point2, width, height, delay = 0.01): 
        "Initializes a visualization with the specified parameters."
        # Number of seconds to pause after each frame
        self.delay = delay

        self.max_dim = max(width, height)
        self.width = width
        self.height = height
        self.point1 = point1
        self.point2 = point2

        # Initialize a drawing surface
        self.master = Tk()
        self.w = Canvas(self.master, width=500, height=500)
        self.w.pack()
        self.master.update()
        
        # Draw a backing and lines
        x1, y1 = self._map_coords(0, 0)
        x2, y2 = self._map_coords(width, height)
        self.w.create_rectangle(x1, y1, x2, y2, fill = "white")

        self.time = 0
        self.master.update()

    ## Maps grid positions to window positions (in pixels)
    def _map_coords(self, x, y):
        return (252 + 500 * ((x - self.width / 2.0) / self.max_dim), 252 + 500 * ((self.height / 2.0 - y) / self.max_dim))

    ## Returns 2 polygons for the 2 points
    def _draw_point(self, point1, point2):
        x, y = point1[0], point1[1]
        d1 = point1[2] + 70
        d2 = point1[2] - 70
        x1, y1 = self._map_coords(x, y)
        x2, y2 = self._map_coords(x + 2.0 * math.sin(d1), y + 2.0 * math.cos(d1))
        x3, y3 = self._map_coords(x + 2.0 * math.sin(d2), y + 2.0 * math.cos(d2))
        a, b = point2[0], point2[1]
        e1 = point2[2] + 70
        e2 = point2[2] - 70
        a1, b1 = self._map_coords(a, b)
        a2, b2 = self._map_coords(a + 2.0 * math.sin(e1), b + 2.0 * math.cos(e1))
        a3, b3 = self._map_coords(a + 2.0 * math.sin(e2), b + 2.0 * math.cos(e2))
        return self.w.create_polygon([x1, y1, x2, y2, x3, y3], fill="blue"), self.w.create_polygon([a1, b1, a2, b2, a3, b3], fill="red")

    ## Redraws the visualization
    def update(self, point1, point2, width, height):
        x1, y1 = self._map_coords(0, 0)
        x2, y2 = self._map_coords(width, height)
        self.w.create_rectangle(x1, y1, x2, y2, fill = "white")
        
        # Draw new points
        self.pointsList = []
        x, y = point1[0], point1[1]
        x1, y1 = self._map_coords(x - 0.1, y - 0.1)
        x2, y2 = self._map_coords(x + 0.1, y + 0.1)
        self.pointsList.append(self.w.create_oval(x1, y1, x2, y2, fill = "black"))
        x, y = point2[0], point2[1]
        x1, y1 = self._map_coords(x - 0.1, y - 0.1)
        x2, y2 = self._map_coords(x + 0.1, y + 0.1)
        self.pointsList.append(self.w.create_oval(x1, y1, x2, y2, fill = "black"))
        self.pointsList.append(self._draw_point(point1, point2))

        self.pointsList.append(self._draw_point(point1, point2))
        self.master.update()
        time.sleep(self.delay)

    def done(self):
        "allows the user to close the window."
        mainloop()

