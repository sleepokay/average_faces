import os
import cv2
import numpy as np
import math
import sys


def readPoints(path):
    pointsArray = []
    
    for filePath in sorted(os.listdir(path)):

        if filePath.endswith(".txt"):
            print(filePath)
            points = []            
            
            with open(os.path.join(path, filePath)) as file :
                for line in file :
                    x, y = line.split()
                    points.append((int(x), int(y)))
            pointsArray.append(points)
            
    return pointsArray

def readImages(path):
    
    imagesArray = []
    
    for filePath in sorted(os.listdir(path)):
        if filePath.endswith(".jpg"):
            print(filePath)
            img = cv2.imread(os.path.join(path,filePath))

            # convert to float32 representation for cv2
            img = np.float32(img)/255.0

            imagesArray.append(img)
            
    return imagesArray
                
# given two sets of two points, compute similarity transform
# openCV requires three pairs of corresponding points we will give a dummy pair for the third
def similarityTransform(inPoints, outPoints):
    s60 = math.sin(60*math.pi/180)
    c60 = math.cos(60*math.pi/180)  
  
    inPts = np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()
    
    xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
    yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]
    
    inPts.append([np.int(xin), np.int(yin)])
    
    xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
    yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]
    
    outPts.append([np.int(xout), np.int(yout)])
    
    tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False)
    
    return tform


# check if a point is inside a rectangle
def rectContains(rect, point):
    if point[0] < rect[0] or point[1] < rect[1] or point[0] > rect[2] or point[1] > rect[3]:
        return False
    return True

# calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    # create subdivision
    subdiv = cv2.Subdiv2D(rect)
   
    for p in points:
        subdiv.insert((p[0], p[1]))

    # get indices of triangles in the points array
    triangleList = subdiv.getTriangleList()
    delaunayTri = []
    
    for t in triangleList:
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            for j in xrange(0, 3):
                for k in xrange(0, len(points)):
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))
    return delaunayTri


def constrainPoint(p, w, h):
    p = (min(max(p[0], 0), w-1), min(max(p[1], 0), h-1))
    return p

# apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # find the affine transform for a pair of triangles
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # then apply the affine transform to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2):

    # find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in xrange(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    # get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # apply warpImage to rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]   
    size = (r2[2], r2[3])
    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    img2Rect = img2Rect * mask

    # copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect



if __name__ == '__main__':
    
    # the file processing is really clunky and hacky, refactor in the future
    path = 'output/'

    dirs = os.walk(path).next()[1]

    for d in dirs:
        print(d)

        # dimensions of output image
        w = 600
        h = 600

        allPoints = readPoints(path + "/" + d + "/")
        images = readImages(path + "/" + d + "/")
        
        eyecornerDst = [(np.int(0.3*w), np.int(h/3)), (np.int(0.7*w ), np.int(h/3))]
        
        imagesNorm = []
        pointsNorm = []
        
        # add boundary points for delaunay triangulation
        boundaryPts = np.array([(0,0), (w/2,0), (w-1,0), (w-1,h/2), ( w-1, h-1 ), ( w/2, h-1 ), (0, h-1), (0,h/2) ])
        
        # initialize location of average points to 0s
        pointsAvg = np.array([(0,0)]* ( len(allPoints[0]) + len(boundaryPts) ), np.float32())
        
        n = len(allPoints[0])

        numImages = len(images)
        
        # warp images and trasnform landmarks to target coordinate system
        # and find average of transformed landmarks.
        
        for i in xrange(0, numImages):

            points1 = allPoints[i]

            # corners of the eye in input image
            eyecornerSrc  = [ allPoints[i][36], allPoints[i][45] ] 
            
            # compute similarity transform
            tform = similarityTransform(eyecornerSrc, eyecornerDst)
            
            # apply similarity transformation
            img = cv2.warpAffine(images[i], tform, (w,h))

            # apply similarity transform on points
            points2 = np.reshape(np.array(points1), (68,1,2))        
            
            points = cv2.transform(points2, tform)
            
            points = np.float32(np.reshape(points, (68, 2)))
            
            # append boundary points to be used in delaunay triangulation
            points = np.append(points, boundaryPts, axis=0)
            
            # calculate location of average landmark points.
            pointsAvg = pointsAvg + points / numImages
            
            pointsNorm.append(points)
            imagesNorm.append(img)
                
        # delaunay triangulation
        rect = (0, 0, w, h)
        dt = calculateDelaunayTriangles(rect, np.array(pointsAvg))

        # output image landmarks
        output = np.zeros((h,w,3), np.float32())

        # warp input images to average image landmarks
        for i in xrange(0, len(imagesNorm)) :
            img = np.zeros((h,w,3), np.float32())
            # Transform triangles one by one
            for j in xrange(0, len(dt)) :
                tin = [] 
                tout = []
                
                for k in xrange(0, 3) :                
                    pIn = pointsNorm[i][dt[j][k]]
                    pIn = constrainPoint(pIn, w, h)
                    
                    pOut = pointsAvg[dt[j][k]]
                    pOut = constrainPoint(pOut, w, h)
                    
                    tin.append(pIn)
                    tout.append(pOut)                
                
                warpTriangle(imagesNorm[i], img, tin, tout)

            # add image intensities for averaging
            output = output + img
        # divide to get average
        output = output / numImages

        # output: imwrite requires uint8 instead of float32
        output *= (255/output.max())    
        output = output.astype(np.uint8)
        cv2.imwrite(str(d) + "-average.jpg", output)