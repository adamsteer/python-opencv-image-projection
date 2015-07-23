# Reverse photography

##h3D-II sensor size
# 36 * 48 mm, 0.036 x 0.048m
## focal length
# 28mm, 0.028m
## multiplier
# 1.0


from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.spatial import distance
import shapefile as shp


def buildshape(corners, filename):
    """build a shapefile geometry from the vertices of the image in 
       world coordinates, then save it using the image name. Sub critical"""
      
    #create a shapefile instance
    #shape = shp.writer(shape.POLYGON)
    #shape.poly(parts = [[proj_coords[:,0], proj_coords[:,1]], [proj_coords[:,1], proj_coords[:,2]]
    #                                  [proj_coords[:,3], proj_coords[:,2]], [proj_coords[:,0], proj_coords[:,3]]]
    #shape.save("./", filename)

def worldfile(corners, im_pix, filename, filepath):
    """build a world file from the vertices of the image in 
       world coordinates, then save it using the image name.
       here we build a small array and then dump it to a file
       
       input is:
       - the image file name       
       - projected corners in world coordinates (*not* bounding box)
       - pxel resolution as a two-element vector [pix_x, pix_y]
       - path to warped image files
       
       reference:
       http://support.esri.com/en/knowledgebase/techarticles/detail/17489
    """
    world_arr = np.zeros([6,1])
    #line 1 is the X pixel resolution in M
    world_arr[0] = im_pix[0]
    #line 2 is the Y pixel resolution in M
    world_arr[3] = -im_pix[1]
    #now the X coord of the top left corner
    world_arr[4] = np.min(corners[0,:])
    #and the Y coordinate of the top left corner
    world_arr[5] = np.max(corners[1,:])
    
    #strip some parts from the filename
    filename = filename[0:len(filename)-4]    
    
    np.savetxt(filepath + filename + '.jpgw', world_arr, "%.3f")
    
    
    

#------
# 2D homogeneous vectors and transformations
def hom2(x, y):
    """2D homogeneous column vector."""
    return np.matrix([x, y, 1]).T

def scale2d(s_x, s_y):
    """Scale matrix that scales 2D homogeneous coordinates"""
    return np.matrix([[s_x, 0, 0],
		     [0, s_y, 0],
		     [0, 0,   1]] )

def trans2d(t_x, t_y):
    """Translation matrix that moves a (homogeneous) vector [v_x, v_y, 1]
       to [v_x + t_x, v_y + t_y, 1]"""
    return np.matrix([[1, 0, t_x],
		     [0, 1, t_y],
		     [0, 0, 1]] )


#-----
# 3D homogeneous vectors and transformations
def hom3(x, y, z):
    """3D homogeneous column vector."""
    return np.matrix([x, y, z, 1]).T

def unhom(v):
    """Convert homogeneous coords (v_x, v_y, v_z, v_w) to 3D by
    (v_x, v_y, v_z) / v_w."""
    return v[:-1]/v[-1]

def trans3d(t):
    """Translation matrix that moves a (homogeneous) vector [v_x, v_y, v_z, 1]
       to [v_x + t_x, v_y + t_y, v_z + t_z, 1]."""
    I = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    return np.hstack([I, t]) 
    # return np.matrix([[1, 0, 0, t_x],
		     # [0, 1, 0, t_y],
		     # [0, 0, 1, t_z],
		     # [0, 0, 0, 1  ]] )
		   

def persp3d(v_x, v_y, v_z):
    """Perspective transformation in homogeneous coordinates
    where v represents the viewer's position relative to the 
    display surface (i.e., (v_x, v_y) is centre, v_z is focal
    distance."""
    return np.matrix([[1, 0, -v_x/v_z, 0],
		     [0, 1, -v_y/v_z, 0],
		     [0, 0,     1,    0],
		     [0, 0,    1/v_z, 0]] )
		   
def cross(a, b):
    """Compute 3D cross product of homogeneous vectors, returning
    the result as a 3D column vector."""
    a3, b3 = unhom(a), unhom(b)
    return np.matrix(np.cross(a3.T, b3.T)).T
    
# Compute zero point in hom. coords
ZERO = hom3(0,0,0)

#-------
# Homogeneous lines, planes, and their intersections
# Based on the discussion here: 
# http://math.stackexchange.com/questions/400268/equation-for-a-line-through-a-plane-in-homogeneous-coordinates

def line(v, x):
    """A homoegeneous line with direction v through a point x is the pair
       (v, x `cross` v)."""
    return np.vstack([unhom(v), cross(x, v)])

def plane(n, x):
    """A plane with normal n passing through x is represented homogeneously 
    by (n, -x.n)."""
    n3 = unhom(n)
    x3 = unhom(x)
    return np.vstack([n3, -(x3.T * n3)])

def meet(P):
    """Compute the meet operator for the given plane W."""
    n, d = P[:3], P[3].item(0,0)
    nx = np.matrix([[0,      -n[2], -n[1]], 
                    [n[2],   0,     -n[0]],
                    [-n[1],  n[0],  0]])
    left = np.vstack([np.diag([-d, -d, -d]), n.T])
    right = np.vstack([nx, np.matrix([0, 0, 0])])
    return np.hstack([left, right])

def intersect(L, P):
    """Compute the point of intersection between the line L and plane P.
    Returned point is homogenous."""

    return meet(P) * L


#-------
# Camera
class Attitude:

    def __init__(self, heading, pitch, roll):
        """Construct a new attitude. Input in degrees, stored in radians.
           ADS: adjusted heading by 180 to keep corner procession intact. 
           TL_im -> TL_grnd, TR_im -> TR_gnd, BR_im -> BR_gnd, BL_im -> BR_gnd 
        """
        self.heading = heading * np.pi / 180.0
        self.pitch   = pitch   * np.pi / 180.0
        self.roll    = roll    * np.pi / 180.0

    def rotation(self):
        """4 x 4 rotation matrix for 3D hom. vectors for this attitude."""
        heading, pitch, roll = self.heading, self.pitch, self.roll

        RX = np.matrix( 
            [[1,	    0,		    0,		    0],
             [0,	    np.cos(pitch),   -np.sin(pitch),  0],
             [0,	    np.sin(pitch),   np.cos(pitch),   0],
             [0,        0,              0,		    1]] )

        RY = np.matrix(
            [[np.cos(roll),    0,	    np.sin(roll),  0],
             [0,		    1,      0,		    0],
             [-np.sin(roll),   0,	    np.cos(roll),  0],
             [0,		    0,      0,		    1]] )
             
        RZ = np.matrix( 
            [[np.cos(heading),  -np.sin(heading),	0, 0],
             [np.sin(heading),  np.cos(heading),	0, 0],
             [0,		    0,			1, 0],
             [0,		    0,			0, 1]] )


        return RZ * RY * RX


"""original rotations

        RX = np.matrix( 
            [[1,	    0,		    0,		    0],
             [0,	    np.cos(roll),   -np.sin(roll),  0],
             [0,	    np.sin(roll),   np.cos(roll),   0],
             [0,        0,              0,		    1]] )

        RY = np.matrix(
            [[np.cos(pitch),    0,	    np.sin(pitch),  0],
             [0,		    1,      0,		    0],
             [-np.sin(pitch),   0,	    np.cos(pitch),  0],
             [0,		    0,      0,		    1]] )
"""


class Sensor:

    def __init__(self, pixel_dim, sensor_dim, focal_length):
        """New sensor of given focal length, sensor dimensions (width, height)
        in metres, and pixel dimensions (width, height)."""
        self.pixels = pixel_dim
        self.screen = sensor_dim
        self.f = focal_length

    def pixel_to_screen(self):
        """Returns a 3x3 matrix transformation that takes 2D hom. pixel 
        coordinates to 2D hom. sensor coordinates."""
        px,sc = self.pixels, self.screen
        T_centre = trans2d(-px[0]/2, -px[1]/2)
        T_scale = scale2d(sc[0]/px[0], sc[1]/px[1])
        return T_scale * T_centre

    def fov_angle(self):
        """Get the FOV angle for this sensor."""
        return 2 * np.arctan(self.w / (2 * self.f))


class Camera:

    def __init__(self, position, attitude, sensor):
        self.position = position
        self.attitude = attitude
        self.sensor = sensor

    def pix_to_world(self):
        """Compute the matrix transform from image to world coordinates.
        Returns a 4 x 3 matrix that converts 2D hom. coords to 3D hom. coords."""

        T_px_to_sc = self.sensor.pixel_to_screen()
        T_2d3d = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0],
                           [0, 0, 1]],
                          dtype = 'float64' )
        T_trans_s = trans3d(hom3(0, 0, self.sensor.f))
        T_att = self.attitude.rotation()
        T_trans_w = trans3d(self.position)

        return T_trans_w * T_att * T_trans_s * T_2d3d * T_px_to_sc 

    def project(self, r, P):
        """Calculate (unhom.) 3D point on plane P that corresponds to pixel 
        coordinate given in hom. 2D coords by r."""

        # Contruct a line through pixel position in world and camera in world
        r_w = self.pix_to_world() * r
        v = ZERO + self.position - r_w
        L = line(v, r_w)
        
        # Get location of project on surface of plane 
        return unhom(intersect(L, P))



#===========

# Set up example camera and ground plane
# Camera sensor is 640 x 480 pixel sensor that is 48mm x 36mm with focal length
# of 28mm and is located at (1000, 1000, 100) with a pitch of -15 degrees.

#
#test data for image 20121023_f13_0044.jpg
# local XYZ
#74.81761600 -55.15724800 303.97706400
#HPR
#306.977 -3.119 1.668

#need to know LiDAR mean range for the flight - let's say it is -30m
# relative to the ellipsoid... so we add that to aircraft Z.

xpix = 8176
ypix = 6132
sensor_x = 0.048
sensor_y = 0.036
focal_len = 0.028

# Set up corners of the image in pixel coordinates
botleft = hom2(0, 6132)
topleft = hom2(0, 0)
botright = hom2(8176, 6132)
topright =hom2(8176, 0)

raw_coords = np.hstack([topleft, topright, botright, botleft])
print("Pixel Coordinates:\n{}".format(raw_coords))

# Ground plane is z=0
ground = plane(hom3(0,0,1), hom3(0,0,0))

im_dir = '../SIPEX2_f9_test/'

trajectory_file = '../SIPEX2_f9_test/20121003_f9_survey_testims_utm.txt'
imlist = '../SIPEX2_f9_test/imnames.txt'

lidar_z = -2.5
# this will be replaced by an estimate for each camera centre...

cameracentres = np.genfromtxt(trajectory_file)

#apply any boresight misalignment information...
# from the 2012 camberidge calibration flight.
#can be determined empirically if you have time!
h_adj = -1.93
p_adj = 1.8292
r_adj = 1.2262

cameracentres[:,6] = cameracentres[:,6] + h_adj  
cameracentres[:,5] = cameracentres[:,5] + p_adj  
cameracentres[:,4] = cameracentres[:,4] + r_adj  


with open(imlist) as f:
    image_list = f.read().splitlines()

i = 0
#now to time-dependent things...
for image in image_list:
    flight_x = cameracentres[i,1]
    flight_y = cameracentres[i,2]
    flight_z = cameracentres[i,3]
    
    flight_h = cameracentres[i,6]
    flight_p = cameracentres[i,5]
    flight_r = cameracentres[i,4]
    
    range_to_ground = flight_z - lidar_z;
    print("camera E:\n{}".format(flight_x))
    print("camera N:\n{}".format(flight_y))
    print("camera U:\n{}".format(range_to_ground))
    print("camera H:\n{}".format(flight_h))
    print("camera P:\n{}".format(flight_p))
    print("camera R:\n{}".format(flight_r))
    
    camera = Camera(
        hom3(flight_x, flight_y, range_to_ground), 
        Attitude(flight_h - 180, flight_p, flight_r),
        Sensor((xpix, ypix), (sensor_x, sensor_y), focal_len))
    
    proj_coords = np.hstack([
        camera.project(topleft, ground), camera.project(topright, ground),
        camera.project(botright, ground), camera.project(botleft, ground)
    ])
    
    print("Ground Coordinates:\n{}".format(proj_coords))
    
    ##now we have some ground coordinates to make projection data, we
    #make a display image
    # a translation and scale. First, scale
    
    #length of each side...
    
    toplen = distance.euclidean(proj_coords[:,0], proj_coords[:,1])
    rightlen = distance.euclidean(proj_coords[:,1], proj_coords[:,2])
    botlen =  distance.euclidean(proj_coords[:,3], proj_coords[:,2])
    leftlen = distance.euclidean(proj_coords[:,0], proj_coords[:,3])
    
    worldres_top = toplen/xpix
    worldres_bot = botlen/xpix
    worldres_x = np.mean([worldres_top, worldres_bot])
    
    print("mean X pixel resolution:\n{}".format(worldres_x))
    
    worldres_left = leftlen/ypix
    worldres_right = rightlen/ypix
    worldres_y = np.mean([worldres_left, worldres_right])
    
    print("mean Y pixel resolution:\n{}".format(worldres_y))
    
    #display_coords
    
    world_bbox_x = np.ceil(np.max(proj_coords[0,:]) - np.abs(np.min(proj_coords[0,:])))
    world_bbox_y = np.ceil(np.max(proj_coords[1,:]) - np.abs(np.min(proj_coords[1,:])))
    
    print("world bbox X:\n{}".format(world_bbox_x))
    print("world bbox Y:\n{}".format(world_bbox_y))
    
    pix_bbox_x = np.ceil(world_bbox_x / worldres_x)
    pix_bbox_y = np.ceil(world_bbox_y / worldres_y)
    
    print("pixel bbox X:\n{}".format(pix_bbox_x))
    print("pixel bbox Y:\n{}".format(pix_bbox_y))
    
    
    ##problem here - we're not projecting from the centroid of a bounding box...  
    ##so we start with an arbitrary x/y
    camera2 = Camera(
        #hom3(pix_bbox_x/2, pix_bbox_y/2, flight_z/worldres_x), 
        hom3(10000, 10000, flight_z/worldres_x),
        Attitude(flight_h - 180, flight_p, flight_r),
        Sensor((xpix, ypix), (sensor_x, sensor_y), focal_len))
    
    im_plot_coords = np.hstack([
        camera2.project(topleft, ground), camera2.project(topright, ground),
        camera2.project(botright, ground), camera2.project(botleft, ground)
        ])
    
    ##and compute a bounding box from the pixel dimensions in im_pot_coords
    pix_bbox_x = np.max(np.ceil(im_plot_coords[0,:])) - np.min(np.floor(im_plot_coords[0,:]))
    pix_bbox_y = np.max(np.ceil(im_plot_coords[1,:])) - np.min(np.floor(im_plot_coords[1,:]))

    print("pixel bbox X:\n{}".format(pix_bbox_x))
    print("pixel bbox Y:\n{}".format(pix_bbox_y)) 
    
    #from_coords = raw_coords[0:2,:]
    from_coords = np.float32(raw_coords[0:2,:])

    
    #to_coords = im_plot_coords[0:2,:]
    # a hah! we need to reset these so that uppermost Y = 0, leftmost X = 0...
    
    #lets try...
    trans_coords = np.zeros_like(im_plot_coords)
    trans_coords[0,:] = im_plot_coords[0,:] - np.min(im_plot_coords[0,:])
    trans_coords[1,:] = im_plot_coords[1,:] - np.min(im_plot_coords[1,:])   
    
    to_coords = np.float32(trans_coords[0:2,:]) 
    #to_coords = np.float32(im_plot_coords[0:2,:]) 
 
  
    img = io.imread(im_dir + image)
    
    p_tform = cv2.getPerspectiveTransform(from_coords.T, to_coords.T)

    img_rot = cv2.warpPerspective(img, p_tform, (np.int(pix_bbox_x), np.int(pix_bbox_y)), cv2.INTER_LANCZOS4)
    
    #img_rot = cv2.warpPerspective(img, p_tform, (np.int(pix_bbox_x), np.int(pix_bbox_y)))
    
    im_pix = img_rot.shape
    #punch out a little world file for warped images    
    worldfile(proj_coords, (worldres_x, worldres_y), image, '../warped/')
    
    #f, (ax0, ax1) = plt.subplots(1, 2)
    #ax0.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    #ax1.imshow(img_rot, cmap=plt.cm.gray, interpolation='nearest')
    #plt.show()
    
    cv2.imwrite('../warped/'+image, img_rot)
    i = i+1

