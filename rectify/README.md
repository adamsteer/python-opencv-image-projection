# warpng images based on camera position/orientation

the bits of code do a couple of jobs:

1. use lens parameters to undistort an image. Lens parameters need to come from something which generates them (eg agisoft photoscan), and this has an openCV dependency.
2. use openCVs projection transform to warp an image from rectangular to it's place in some cartesian reference system

## notes

This stuff was designed as a quick and dirty image warper for some research work I did on sea ice. it assumes that there is no terrain, and relies on really good camera positioning.

## dependencies

The biggest headache is openCV.
It also relies on scikits-image and scipy.spatial

Better documentation to come! And maybe better code...
