import numpy as np
import matplotlib.pyplot as plt
import timeit

#Question 1

#1a

#import time module
from astropy.time import Time
#import coordinate things
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body_barycentric_posvel
#import units
from astropy import units as u

#set the current time
t_now = Time("2021-12-07 10:00")

with solar_system_ephemeris.set('jpl'):
	sun = get_body_barycentric_posvel("sun", t_now)
	mercu = get_body_barycentric_posvel("mercury", t_now)
	venus = get_body_barycentric_posvel("venus", t_now)
	earth = get_body_barycentric_posvel("earth", t_now)
	mars = get_body_barycentric_posvel("mars", t_now)
	jupi = get_body_barycentric_posvel("jupiter", t_now)
	sat = get_body_barycentric_posvel("saturn", t_now)
	uran = get_body_barycentric_posvel("uranus", t_now)
	nept = get_body_barycentric_posvel("neptune", t_now)

sunposition, sunvelocity = sun[0], sun[1]
mercposition, mercvelocity = mercu[0], mercu[1]
venusposition, venusvelocity = venus[0], venus[1]
earthposition, earthvelocity = earth[0], earth[1]
marsposition, marsvelocity = mars[0], mars[1]
jupposition, jupvelocity = jupi[0], jupi[1]
satposition, satvelocity = sat[0], sat[1]
uranposition, uranvelocity = uran[0], uran[1]
neptposition, neptvelocity = nept[0], nept[1]

#Creating an array of positions and velocities
solarsyspos = np.array([sunposition, mercposition, venusposition, earthposition, marsposition, jupposition, satposition, uranposition, neptposition])
solarsysvel = np.array([sunvelocity, mercvelocity, venusvelocity, earthvelocity, marsvelocity, jupvelocity, satvelocity, uranvelocity, neptvelocity])

names = np.array(["sun", "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"])

colors = np.array(["gold", "brown", "grey", "blue", "red", "orange", "pink", "lightsteelblue", "darkblue"])


for i in range(9):
	x = solarsyspos[i].x.to_value(u.AU)
	y = solarsyspos[i].y.to_value(u.AU)
	name = names[i]
	color = colors[i]
	
	#vx = solarsysvel[i].x.to_value(u.AU/u.day)
	#vy = solarsysvel[i].y.to_value(u.AU/u.day)

	plt.scatter(x,y, label=name, color=color)
	#plt.arrow(x,y,vx,vy, color=color)
	
plt.legend()
plt.title("Solar system at time "+str(t_now))
plt.xlabel("x [AU]")
plt.ylabel("y [AU]")
plt.savefig('SolarSystemPresentxy.png')
plt.close()

for i in range(9):
	x = solarsyspos[i].x.to_value(u.AU)
	z = solarsyspos[i].z.to_value(u.AU)
	name = names[i]
	color = colors[i]

	#vx = solarsysvel[i].x.to_value(u.AU/u.day)
	#vz = solarsysvel[i].z.to_value(u.AU/u.day)

	plt.scatter(x,z, label=name, color=color)
	#plt.arrow(x,z,vx,vz, color=color)
	
plt.legend()
plt.title("Solar system at time "+str(t_now))
plt.xlabel("x [AU]")
plt.ylabel("z [AU]")
plt.savefig('SolarSystemPresentxz.png')
plt.close()
	
#1b (and 1c)

from astropy import constants as c

Grav = c.G.to_value()
Msun = c.M_sun.to_value()

#Import function to calculate the gravitational acceleration from the tutorial
def gravity(t,r):
    """Assumes r = [x,y,z] and is in units of m and calculates the acceleration"""
    denom = 1/(r[0]**2 + r[1]**2 + r[2]**2)**(3/2)
    a = -(Grav * Msun * r * denom)
    #a_y = -(c.G * c.M_sun * r[1]*u.m * denom).to_value()
    #a_z = -(c.G * c.M_sun * r[2]*u.m * denom).to_value()
    
    #a = np.array([a_x, a_y, a_z])
    
    return a

#Move everything such that the sun is at [0,0,0] and stays there
for i in range(9):
	solarsyspos[i] = solarsyspos[i] - solarsyspos[0]

#A year and a day in seconds
year_sec = 3600 * 24 * 365
day_sec = 3600 * 24

#Import the leap frog function from the tutorial (but now for 3D)
def leapfrog(func,start,stop,h,y0,z0):
    """Integrates a second order ODE based on the leap frog method, given an acceleration func, a starting point start, ending point stop, stepsize h, initial values for the position y0 and initial values for the speed z0."""
    #time
    times = np.arange(start,stop+h,h)
    #positions
    positions = np.zeros((len(times), 3))
    #initial position
    positions[0,:] = y0
    #velocities
    velocities = np.zeros((len(times), 3))
    #initial velocity
    velocities[0,:] = z0
    
    #looping over time steps
    for i in range(1,len(times)):
        #set intial v_1/2
        k1 = func(times[i-1],positions[i-1,:])*h
        if i == 1:
            velocities[i,:] = velocities[i-1,:] + 0.5*k1
        #calculate v_i+1/2
        else:
            velocities[i,:] = velocities[i-1,:] + k1
        
        #calculate x_i+1 by taking v_i+1/2 * stepsize
        positions[i,:] = positions[i-1,:] + velocities[i,:]*h
        
    return times, positions, velocities

def Euler(func,start,stop,h,y0,z0):
    """Integrates a second order ODE based on Euler's method, given an acceleration func, a starting point start, ending point stop, stepsize h, initial values for the position y0 and initial values for the speed z0."""
    #time
    times = np.arange(start,stop+h,h)
    #positions
    positions = np.zeros((len(times), 3))
    #initial position
    positions[0,:] = y0
    #velocities
    velocities = np.zeros((len(times), 3))
    #initial velocity
    velocities[0,:] = z0
    
    #looping over time steps
    for i in range(1,len(times)):
        k1 = func(times[i-1],positions[i-1,:])*h
        velocities[i,:] = velocities[i-1,:] + k1
        #calculate x_i+1 by taking v_i * stepsize
        positions[i,:] = positions[i-1,:] + velocities[i-1,:]*h
        
    return times, positions, velocities
    


figure = plt.figure(figsize=(10,10))
times = np.arange(0, 200*year_sec+0.5*day_sec, 0.5*day_sec)
N = len(times)

#Positions of the planets for Leap Frog
xpositionsLF = np.zeros((8,N))
ypositionsLF = np.zeros((8,N))
zpositionsLF = np.zeros((8,N))

#Positions of the planets for Euler
xpositionsE = np.zeros((8,N))
ypositionsE = np.zeros((8,N))
zpositionsE = np.zeros((8,N))

for i in range(9):
	#Taking position in meters
	x = solarsyspos[i].x.to_value(u.m)
	y = solarsyspos[i].y.to_value(u.m)
	z = solarsyspos[i].z.to_value(u.m)

	name = names[i]
	color = colors[i]
	
	if i == 0:
		plt.scatter(x,y, label=name, color=color)

	if i > 0:
		#Making an r array with "dimensionless" x,y,z
		r = np.array([x, y, z])

		#Taking velocity in m/s
		vx = solarsysvel[i].x.to_value(u.m/u.s)
		vy = solarsysvel[i].y.to_value(u.m/u.s)
		vz = solarsysvel[i].z.to_value(u.m/u.s)

		#Making an velocity array with "dimensionless" vx,vy,vz
		vels = np.array([vx, vy, vz])
		
		#Doing the leapfrog integration
		ts, rs, vs = leapfrog(gravity, 0, 200*year_sec, 0.5*day_sec, r, vels)
		#And Euler
		tsE, rsE, vsE = Euler(gravity, 0, 200*year_sec, 0.5*day_sec, r, vels)

		#Saving the positions in AU
		xpositionsLF[i-1,:] = (rs[:,0]*u.m).to_value(u.AU)
		ypositionsLF[i-1,:] = (rs[:,1]*u.m).to_value(u.AU)
		zpositionsLF[i-1,:] = (rs[:,2]*u.m).to_value(u.AU)
		
		xpositionsE[i-1,:] = (rsE[:,0]*u.m).to_value(u.AU)
		ypositionsE[i-1,:] = (rsE[:,1]*u.m).to_value(u.AU)
		zpositionsE[i-1,:] = (rsE[:,2]*u.m).to_value(u.AU)

		plt.plot(xpositionsLF[i-1,:], ypositionsLF[i-1,:], color=color, label=name)


plt.legend()
plt.title("Solar system at "+str(t_now)+" + 200 years into the future (LeapFrog)")
plt.xlabel("x [AU]")
plt.ylabel("y [AU]")
plt.savefig('SolarSystemLeapFrogxy.png')
plt.close()

figure = plt.figure(figsize=(16,10))
for i in range(9):
	#Taking position in meters

	name = names[i]
	color = colors[i]
	
	if i == 0:
		z = solarsyspos[i].z.to_value(u.AU)
		plt.plot(times,z*np.ones(N), label=name, color=color)

	if i > 0:
		plt.plot(times, zpositionsLF[i-1,:], color=color, label=name)

plt.legend()
plt.title("Solar system at "+str(t_now)+" + 200 years into the future (LeapFrog)")
plt.xlabel("t [s]")
plt.ylabel("z [AU]")
plt.savefig('SolarSystemLeapFrogtz.png')
plt.close()

figure = plt.figure(figsize=(10,10))
for i in range(9):
	#Taking position in meters

	name = names[i]
	color = colors[i]
	
	if i == 0:
		x = solarsyspos[i].x.to_value(u.AU)
		y = solarsyspos[i].y.to_value(u.AU)
		plt.scatter(x,y, label=name, color=color)

	if i > 0:
		plt.plot(xpositionsE[i-1,:], ypositionsE[i-1,:], color=color, label=name)

plt.legend()
plt.title("Solar system at "+str(t_now)+" + 200 years into the future (Euler)")
plt.xlabel("x [AU]")
plt.ylabel("y [AU]")
plt.savefig('SolarSystemEulerxy.png')
plt.close()

figure = plt.figure(figsize=(16,10))
for i in range(9):
	#Taking position in meters

	name = names[i]
	color = colors[i]
	
	if i == 0:
		z = solarsyspos[i].z.to_value(u.m)
		plt.plot(times,z*np.ones(N), label=name, color=color)

	if i > 0:
		plt.plot(times, zpositionsE[i-1,:], color=color, label=name)

plt.legend()
plt.title("Solar system at "+str(t_now)+" + 200 years into the future (Euler)")
plt.xlabel("t [s]")
plt.ylabel("z [AU]")
plt.savefig('SolarSystemEulertz.png')
plt.close()

figure = plt.figure(figsize=(10,10))
for i in range(1,9):
	#Taking position in meters

	name = names[i]
	color = colors[i]

	if i > 0:
		plt.plot(times, np.abs(xpositionsLF[i-1,:]-xpositionsE[i-1,:]), color=color, label=name)

plt.legend()
plt.title("Solar system at "+str(t_now)+" + 200 years into the future (Euler)")
plt.xlabel("t [s]")
plt.ylabel("difference in x position Euler vs Leapfrog [AU]")
plt.savefig('SolarSystemEulerLeapFrogDiff.png')
plt.close()








