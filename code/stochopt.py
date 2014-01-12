import math
import random
from Tkinter import *
import stochoptVisuals

## function to get angle diff
def angleDiff(angle1, angle2): # theta and phi are the current angles of the 2 pts
    return abs(angle1 - angle2)

## function to get distance diff, using sum of coordinates-squared
def dDiff(x1, y1, x2, y2):
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

## Write cost function of the 2 pts
## Cost is minimum if the 2 pts face each other, distance d apart
## Cost = Wd*cost(d) + Wa*cost(angle) = Wd*sum(delta-d) + Wa*sum(delta-angle)
def costFunction(point1, point2, d):
    currentAngle = angleDiff(point1[2], point2[2])
    angle = math.pi # pi = 180
    costA = abs(currentAngle - angle)
    currentD = dDiff(point1[0], point1[1], point2[0], point2[1])
    costD = abs(currentD - d)
    # weightings Wd = [1.0, 5.0]; Wa = 10.0 
    cost = 5*(10.0*costA + 5.0*costD) # 5* to make temperature probability smaller 
    return cost                       # later, else will have too many acceptances

# translate a point by dist along front dir
def goForward(point, dist, theta):
    x, y = point[0], point[1]
    angle = math.degrees(theta)
    # for each of the 4 quadrants ASTC
    if angle >= 0 and angle < 90 or angle == 360: #360 == 0
        x += dist*math.cos(theta)
        y += dist*math.sin(theta)
    elif angle >= 90 and angle < 180:
        x -= dist*math.cos(theta-(0.5*math.pi))
        y += dist*math.sin(theta-(0.5*math.pi))
    elif angle >= 180 and angle < 270:
        x -= dist*math.cos(theta-math.pi)
        y -= dist*math.sin(theta-math.pi)
    elif angle >= 270 and angle < 360:
        x += dist*math.cos(theta-(1.5*math.pi))
        y -= dist*math.sin(theta-(1.5*math.pi))
    # to stop points from exceeding screen borders
    if x > 500:
        x = 500
    elif x < 0:
        x = 0
    if y > 500:
        y = 500
    elif y < 0:
        y = 0
    point[0], point[1] = x, y
    return point

# rotate a point by theta in clockwise dir
def rotate(point, theta):
    theta += math.radians(random.randint(0,20))
    return theta

#copy point when for proposal
def copyPt(point):
    return point[:] # creating new list prevents aliasing

# get Proposal Cost Function
def proposalCost(point1, point2, n):
    #so that pt1 and 2 can remain if proposal not taken
    pointA, pointB = copyPt(point1), copyPt(point2)    
    #randomly pick dist and theta, and which pt to use, which fn to use
    if n < 800: # high n so that angle will be closer to n since if n can pick from (0,1) it will almost always result in a better cost function
        dist = float(random.randint(0,10))
    else:
        dist = float(random.random())
    goForwardAngle = math.radians(random.randint(0, 360))
    pickPt, pickFn = random.randint(0, 1), random.randint(0, 1)
    ptList = [pointA, pointB]
    fnList = [goForward(ptList[pickPt], dist, goForwardAngle), rotate(ptList[pickPt], ptList[pickPt][2])]
    if pickFn == 0:
        ptList[pickPt] = fnList[pickFn]
    elif pickFn == 1:
        ptList[pickPt][2] = fnList[pickFn]    
    proposedCost = costFunction(pointA, pointB, d)
    return proposedCost, pointA, pointB

# Acceptance probability due to temperature
def tempAcceptProb(temp, currentCost, proposedCost, minCostFunction):
    beta = float(1.0/temp) ##B is inversely proportional to temp
    acceptanceProbability = math.e**(beta*(minCostFunction - proposedCost))
    if acceptanceProbability > 1: # since max range of probability is 1
        return 1
    return acceptanceProbability

# Print results
def printResults(pointa, pointb):
    print "Finished! point1 = ", pointa, "point2 = ", pointb
    print " result d = ", math.sqrt((pointa[0]-pointb[0])**2 + (pointa[1]-pointb[1])**2)
    angleDiff = math.degrees(abs(pointa[2]-pointb[2]))
    while angleDiff >= 360:
        angleDiff -= 360
    print " angle diff = ", angleDiff
    
##__MAIN FUNCTION_>

d = 0 ## <===--- 
minCostFunction = math.pi + d # min cost function
point1G, point2G = [], []
numIter = 1000 #<=---
temp = 1000
## Set point1; point 2 randomly
width = 100 # hypothetical width of frame
height = 100 # hypothetical height of frame
[x1, y1, angle1] = [float(random.randint(0, width)), float(random.randint(0, height)), math.radians(float(random.randint(0, 360)))]
[x2, y2, angle2] = [float(random.randint(0, width)), float(random.randint(0, height)), math.radians(float(random.randint(0, 360)))]
point1, point2 = [x1, y1, angle1], [x2, y2, angle2]
print "initial point 1 = ", point1, " initial point2 = ", point2

anim = stochoptVisuals.pointVisualization(point1, point2, width, height)

for n in xrange(1, numIter+1):
    # update temperature w every iteration
    temp*=.99
    ## Get current cost
    currentCost = costFunction(point1, point2, d)
    ## Get proposed cost and proposed new point
    proposedCost, pointA, pointB = proposalCost(point1, point2,n)
    acceptanceP = tempAcceptProb(temp, currentCost, proposedCost, minCostFunction)
    if proposedCost == minCostFunction: # minimum cost reached
        point1G, point2G = pointA, pointB
    randomProb = random.random()
    if proposedCost < currentCost or acceptanceP >= 0.9:#randomProb: #lower cost or temp accept probability
        point1, point2 = pointA, pointB
    # else: if neither req are met, don't adopt pointA/pointB
    # draw new frame of the animation
    printResults(point1, point2)
    print "count = ", n
    anim.update(point1, point2, width, height)

# printing results at the end
# if ideal points appeared, then overwrite point1, point2
if point1G == []:
    printResults(point1, point2) 
else:
    printResults(point1G, point2G) 
    point1, point2 = point1G, point2G # overwrite point1, point2
# Draw final points (in case ideal was reached)
anim.update(point1, point2, width, height)
anim.done()

######--------------------------TESTS-------------------------------
"""
count =  1000
Finished! point1 =  [75.120488706612591, 56.886880852998452, 6.3529984772593595] point2 =  [75.109450962426749, 56.811995742151403, 3.2114058236695664]
 result d =  0.0756941980821
 angle diff =  180.0


count =  1000
Finished! point1 =  [71.409206450349416, 75.028971645702242, 6.8242373752978276] point2 =  [71.360382620251144, 75.021930754107757, 3.6651914291880932]
 result d =  0.0493289016694
 angle diff =  181.0

count =  1000
Finished! point1 =  [73.537993544279004, 54.030810187806672, 9.354964790689607] point2 =  [73.590202202383239, 54.048114131025756, 6.213372137099813]
 result d =  0.0550015493598
 angle diff =  180.0
"""
