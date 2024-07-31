#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 20:57:20 2024

@author: johnshreen
"""

import os
import sys
import shelve
import platform
import datetime

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import progressbar
import h5py 

from scipy.integrate import solve_ivp
from scipy.spatial import Voronoi as scipyVoronoi
from scipy.spatial import distance as scipy_distance

projectDir = os.getcwd()
if projectDir != os.getcwd():
    os.chdir(projectDir)
    
import functions_spinning_rafts as fsr

dataDir = os.path.join(projectDir, 'data')
if not os.path.isdir(dataDir):
    os.mkdir('data')

# %% magnetic force and torque calculations
miu0 = 4 * np.pi * 1e-7  # unit: N/A**2, or T.m/A, or H/m

magneticMomentOfOneRaft = 1e-8  # unit: A.m**2

orientationAngles = np.arange(0, 361)  # unit: degree;
orientationAnglesInRad = np.radians(orientationAngles)

    
'''对所有距离都求一遍，相当于构建一个数据库，每次用到不必重复计算'''
magneticDipoleEEDistances = np.arange(0, 10001) / 1e6  # np.arange(0, 10001) / 1e6  # unit: m
radiusOfRaft = 1.5e-4  # unit: m
magneticDipoleCCDistances = magneticDipoleEEDistances + radiusOfRaft * 2  # unit: m



magDpForceOnAxis = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles)))  # unit: N
magDpForceOffAxis = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles)))  # unit: N
magDpTorque = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles)))  # unit: N.m
 

for index, d in enumerate(magneticDipoleCCDistances):
    # magDpEnergy[index, :] = \
    #     miu0 * magneticMomentOfOneRaft ** 2 * (1 - 3 * (np.cos(orientationAnglesInRad) ** 2)) / (4 * np.pi * d ** 3)
    magDpForceOnAxis[index, :] = \
        3 * miu0 * magneticMomentOfOneRaft ** 2 * (1 - 3 * (np.cos(orientationAnglesInRad) ** 2)) / (4 * np.pi * d ** 4)
    magDpForceOffAxis[index, :] = \
        3 * miu0 * magneticMomentOfOneRaft ** 2 * (2 * np.cos(orientationAnglesInRad) *
                                                   np.sin(orientationAnglesInRad)) / (4 * np.pi * d ** 4)
    magDpTorque[index, :] = \
        miu0 * magneticMomentOfOneRaft ** 2 * (3 * np.cos(orientationAnglesInRad) *
                                               np.sin(orientationAnglesInRad)) / (4 * np.pi * d ** 3)
    '''取平均的方法不对吗？  mean还是sum？  应该是mean  
    
    能不能直接取平均呢？这个近似是不是准确？ 肯定不是近似的原因    
    '''

magDpForceOnAxis_average = np.mean(magDpForceOnAxis, axis=1)


# %% plotting to check the forces 
densityOfWater = 1e-15  # unit conversion: 1000 kg/m^3 = 1e-15 kg/um^3
raftRadius = 1.5e2  # unit: micron

'''
这个转速后面被覆盖了，那就注释掉
'''
'''
magneticFieldRotationRPS = 22
omegaBField = magneticFieldRotationRPS * 2 * np.pi
'''

'''这个常数最后要调整到合适为止'''
coefficient_1 = 2e-8

repulsion = coefficient_1/((magneticDipoleCCDistances/(raftRadius/1000000))-1) ** 8  # unit: N

# coefficient_2 = -5e-3

# ForceTangential = coefficient_2/magneticDipoleCCDistances ** 2  # unit: N


'''
这个画图不该出事呀，而且最后也调出可以看到平衡点的图了
反而是看不出平衡点的那一个图像和对应的视频不一致


也不是比例尺计算的问题，我是直接用2R来确定的Distance，根本不涉及像素的测量和计算，
而且用另外的程序测出图像的像素距离来算，照样有矛盾

难道是视频有问题？
视频的大逻辑也不会有错吧，我在引力的前面加负号，合力就表现为斥力，引力前面是正号，
合力就表现为引力，说明力的方向是对的，只不过是引力大于了斥力
只要把斥力的系数放大，就可以实现斥力大于引力。

'''

x = magneticDipoleCCDistances * 1e6
y_1 = magDpForceOnAxis_average
y_2 = repulsion


plt.figure(figsize=(10, 6), dpi=1000)
startDistID = 0
endDistID = 300

plt.plot(x[startDistID:endDistID], y_1[startDistID:endDistID], label='magDpForceOnAxis_average')
plt.plot(x[startDistID:endDistID], y_2[startDistID:endDistID], label='repulsion')
plt.plot(x[startDistID:endDistID], y_1[startDistID:endDistID]+y_2[startDistID:endDistID], label='sum of axial forces')
plt.plot(x[startDistID:endDistID], np.zeros_like(x[startDistID:endDistID]), '--')
plt.xlabel('Distance (micron)')
plt.ylabel('Force (N)')
plt.title('Axial forces')
plt.legend()
plt.show()

# %% Simulation begins 

'''
这个位置给定了浮筏的个数，以及磁场的转速，RPS是从else后面的参数得到的
按照步长为-5，会生成两个视频，RPS=20、25
'''
if len(sys.argv) > 1:
    numOfRafts = int(sys.argv[1])
    spinSpeedStart = int(sys.argv[2])
    spinSpeedStep = -1
    spinSpeedEnd = spinSpeedStart + spinSpeedStep
    
else:
    # 在Spyder中运行时使用默认参数
    numOfRafts = 20 # 默认值
    spinSpeedStart = -20  # 默认值
    spinSpeedStep = -5
    spinSpeedEnd = -30

timeStepSize = 1e-3  # unit: s
numOfTimeSteps = 5000
timeTotal = timeStepSize * numOfTimeSteps


os.chdir(dataDir)
now = datetime.datetime.now()
outputFolderName = now.strftime("%Y-%m-%d") + '_' + str(numOfRafts) + 'Rafts_' + \
                   'timeStep' + str(timeStepSize) + '_total' + str(timeTotal) + 's'

if not os.path.isdir(outputFolderName):
    os.mkdir(outputFolderName)
os.chdir(outputFolderName)


listOfVariablesToSave = ['arenaSize', 'numOfRafts', 'magneticFieldStrength', 'omegaBField',
                         'timeStepSize', 'numOfTimeSteps',
                         'timeTotal', 'outputImageSeq', 'outputVideo', 'outputFrameRate', 'intervalBetweenFrames',
                         'raftLocations', 'raftOrientations', 'raftRadii', 'raftRotationSpeedsInRad',
                         'entropyByNeighborDistances', 'hexaticOrderParameterAvgNorms',
                         'hexaticOrderParameterModuliiAvgs',
                         'deltaR', 'radialRangeArray', 'binEdgesNeighborDistances',
                         ]



# constants of proportionality
cm = 1  # coefficient for the magnetic force term
ch = 1  # coefficient for the hydrodynamic force term
tb = 1  # coefficient for the magnetic field torque term
tm = 1  # coefficient for the magnetic dipole-dipole torque term

unitVectorX = np.array([1, 0])
unitVectorY = np.array([0, 1])

if numOfRafts > 2:
    arenaSize = 1.5e4
else:
    arenaSize = 2e3  # unit: micron
R = raftRadius = 1.5e2  # unit: micron
centerOfArena = np.array([arenaSize / 2, arenaSize / 2])
canvasSizeInPixel = int(1000)  # unit: pixel
scaleBar = arenaSize / canvasSizeInPixel  # unit: micron/pixel


densityOfWater = 1e-15  # unit conversion: 1000 kg/m^3 = 1e-15 kg/um^3
miu = 1e-15  # dynamic viscosity of water; unit conversion: 1e-3 Pa.s = 1e-3 N.s/m^2 = 1e-15 N.s/um^2
piMiuR = np.pi * miu * raftRadius  # unit: N.s/um
magneticFieldStrength = 10e-3  # 10e-3 # unit: T



initialPositionMethod = 2  # 1 -random positions, 2 - fixed initial position,
# 3 - starting positions are the last positions of the previous spin speeds


'''我记得之前有对初始状态下边边距和心心距的设定了，为什么这里又设置一遍呢 已解决'''
ccSeparationStarting = 400  # unit: micron
initialOrientation = 0  # unit: deg
lastPositionOfPreviousSpinSpeeds = np.zeros((numOfRafts, 2))  # come back and check if these three can be removed.
lastOmegaOfPreviousSpinSpeeds = np.zeros(numOfRafts)
firstSpinSpeedFlag = 1




outputImageSeq = 0
outputVideo = 1
outputFrameRate = 100.0
intervalBetweenFrames = int(1)  # unit: steps
blankFrameBGR = np.ones((canvasSizeInPixel, canvasSizeInPixel, 3), dtype='int32') * 255

solverMethod = 'RK45'  # RK45,RK23, Radau, BDF, LSODA



'''先加上切向力，然后画轴向力的图'''

def funct_drdt_dalphadt(t, raft_loc_orient):
    """
    Two sets of ordinary differential equations that define dr/dt and dalpha/dt above and below the threshold value
    for the application of lubrication equations
    """
    #    raft_loc_orient = raftLocationsOrientations
    raft_loc = raft_loc_orient[0: numOfRafts * 2].reshape(numOfRafts, 2)  # in um
    raft_orient = raft_loc_orient[numOfRafts * 2: numOfRafts * 3]  # in deg

    drdt = np.zeros((numOfRafts, 2))  # unit: um
    raft_spin_speeds_in_rads = np.zeros(numOfRafts)  # in rad
    dalphadt = np.zeros(numOfRafts)  # unit: deg
    
    mag_dipole_force_on_axis_term = np.zeros((numOfRafts, 2))
    #hydrodynamic_force_term = np.zeros((numOfRafts, 2))
    mag_dipole_force_off_axis_term = np.zeros((numOfRafts, 2))
    repulsion_term = np.zeros((numOfRafts, 2))
    velocity_torque_coupling_term = np.zeros((numOfRafts, 2))
    
    magnetic_field_torque_term = np.zeros(numOfRafts)
    magnetic_dipole_torque_term = np.zeros(numOfRafts)
    

    # loop for torque
    for raft_id in np.arange(numOfRafts):
        # raft_id = 0
        ri = raft_loc[raft_id, :]  # unit: micron
        # magnetic field torque:
        magnetic_field_torque = \
            magneticFieldStrength * magneticMomentOfOneRaft \
            * np.sin(np.deg2rad(magneticFieldDirection - raft_orient[raft_id]))  # unit: N.m
        magnetic_field_torque_term[raft_id] = tb * magnetic_field_torque * 1e6 / (8 * piMiuR * R ** 2)  # unit: 1/s

        

        for neighbor_id in np.arange(numOfRafts):
            if neighbor_id == raft_id:
                continue
            rj = raft_loc[neighbor_id, :]  # unit: micron
            rji = ri - rj  # unit: micron
            rji_norm = np.sqrt(rji[0] ** 2 + rji[1] ** 2)  # unit: micron
            rji_ee_dist = rji_norm - 2 * R  # unit: micron
            '''下面这个位置，由于除法运算中出现除数为零，报错NAN，所以用了np.divide，但是GitHub中没用，为啥我不用就会报错'''
            rji_unitized = np.divide(rji, rji_norm, out=np.ones_like(rji), where=rji_norm!=0)#rji / rji_norm  # unit: micron
            rji_unitized_cross_z = np.asarray((rji_unitized[1], -rji_unitized[0]))
            phi_ji = (np.arctan2(rji[1], rji[0]) * 180 / np.pi
                      - raft_orient[raft_id]) % 360  # unit: deg; assuming both rafts's orientations are the same

            

            magnetic_dipole_torque_term[raft_id] = magnetic_dipole_torque_term[raft_id] \
                                                       + tm * magDpTorque[int(rji_ee_dist + 0.5), int(phi_ji + 0.5)] \
                                                       * 1e6 / (8 * piMiuR * R ** 2)



        raft_spin_speeds_in_rads[raft_id] = \
            magnetic_field_torque_term[raft_id] + magnetic_dipole_torque_term[raft_id] 
            
    dalphadt = raft_spin_speeds_in_rads / np.pi * 180  # in deg    
        
        
     # loop for forces
    for raft_id in np.arange(numOfRafts):
        # raftID = 0
        ri = raft_loc[raft_id, :]  # unit: micron
        
       # magnetic field torque:
        magnetic_field_torque = \
            magneticFieldStrength * magneticMomentOfOneRaft \
            * np.sin(np.deg2rad(magneticFieldDirection - raft_orient[raft_id]))  # unit: N.m
        magnetic_field_torque_term[raft_id] = tb * magnetic_field_torque * 1e6 / (8 * piMiuR * R ** 2)  # unit: 1/s

        for neighbor_id in np.arange(numOfRafts):
            if neighbor_id == raft_id:
                continue
            rj = raft_loc[neighbor_id, :]  # unit: micron
            rji = ri - rj  # unit: micron
            rji_norm = np.sqrt(rji[0] ** 2 + rji[1] ** 2)  # unit: micron
            rji_ee_dist = rji_norm - 2 * R  # unit: micron
            rji_unitized = np.divide(rji, rji_norm, out=np.ones_like(rji), where=rji_norm!=0)#rji / rji_norm  # unit: micron
            rji_unitized_cross_z = np.asarray((rji_unitized[1], -rji_unitized[0]))
            phi_ji = (np.arctan2(rji[1], rji[0]) * 180 / np.pi - raft_orient[raft_id]) % 360  # unit: deg;
            omegaj = raft_spin_speeds_in_rads[neighbor_id]
        
            mag_dipole_force_on_axis_term[raft_id, :] = mag_dipole_force_on_axis_term[raft_id, :] \
                                                            + cm * magDpForceOnAxis[int(rji_ee_dist + 0.5),
                                                                                    int(phi_ji + 0.5)] \
                                                            * rji_unitized / (6 * piMiuR)  # unit: um/s
                     
            mag_dipole_force_off_axis_term[raft_id, :] = mag_dipole_force_off_axis_term[raft_id, :] \
                                                             + magDpForceOffAxis[int(rji_ee_dist + 0.5),
                                                                                 int(phi_ji + 0.5)] \
                                                             * rji_unitized_cross_z / (6 * piMiuR)
                                                             
                                                             
            repulsion_term[raft_id, :] = repulsion_term[raft_id, :] \
                                        + repulsion[int(rji_ee_dist + 0.5)] * rji_unitized \
                                            / (6 * piMiuR)  # unit: um/s
            
            velocity_torque_coupling_term[raft_id, :] = velocity_torque_coupling_term[raft_id, :] \
                                                        - R ** 3 * omegaj * rji_unitized_cross_z \
                                                        / (rji_norm ** 2)  # unit: um/s

    

        '''每一个力的符号都要确认，到底有没有负号，要是mag_dipole_force_on_axis_term
        能加负号，就能对上结果。    不行，轴向力和斥力都没有负号       
        '''
        '''加上切向力了，可以转起来'''
        drdt[raft_id, :] = \
            mag_dipole_force_on_axis_term[raft_id, :] \
            + repulsion_term[raft_id, :]\
                + mag_dipole_force_off_axis_term[raft_id, :] \
                + velocity_torque_coupling_term[raft_id, :]\

    #dalphadt = raft_spin_speeds_in_rads / np.pi * 180  # in deg

    drdt_dalphadt = np.concatenate((drdt.flatten(), dalphadt))

    return drdt_dalphadt       



'''
这个位置的RPS是根据spinSpeedStart, spinSpeedEnd, spinSpeedStep得到的，已经把之前定义的
RPS=22覆盖掉了
'''

for magneticFieldRotationRPS in np.arange(spinSpeedStart, spinSpeedEnd, spinSpeedStep):
    omegaBField = magneticFieldRotationRPS * 2 * np.pi  # unit: rad/s
    raftLocations = np.zeros((numOfRafts, numOfTimeSteps, 2))  # in microns
    raftOrientations = np.zeros((numOfRafts, numOfTimeSteps))  # in deg
    raftRadii = np.ones(numOfRafts) * raftRadius  # in micron
    raftRotationSpeedsInRad = np.zeros((numOfRafts, numOfTimeSteps))  # in rad
    raftRelativeOrientationInDeg = np.zeros((numOfRafts, numOfRafts, numOfTimeSteps))
    
    
    entropyByNeighborDistances = np.zeros(numOfTimeSteps - 1)
    '''
    hexaticOrderParameterAvgs = np.zeros(numOfTimeSteps, dtype=np.csingle)
    '''
    hexaticOrderParameterAvgNorms = np.zeros(numOfTimeSteps)
    '''
    hexaticOrderParameterMeanSquaredDeviations = np.zeros(numOfTimeSteps, dtype=np.csingle)
    '''
    hexaticOrderParameterModuliiAvgs = np.zeros(numOfTimeSteps)
    hexaticOrderParameterModuliiStds = np.zeros(numOfTimeSteps)


    deltaR = 1
    radialRangeArray = np.arange(2, 100, deltaR)
    binEdgesNeighborDistances = list(np.arange(2, 10, 0.5)) + [100]
    '''
    radialDistributionFunction = np.zeros((numOfTimeSteps, len(radialRangeArray)))  # pair correlation function: g(r)
    
    spatialCorrHexaOrderPara = np.zeros((numOfTimeSteps, len(radialRangeArray)))
    
    # spatial correlation of hexatic order paramter: g6(r)
    spatialCorrHexaBondOrientationOrder = np.zeros((numOfTimeSteps, len(radialRangeArray)))
    # spatial correlation of bond orientation parameter: g6(r)/g(r)
    '''
    
    
    
    
    dfNeighbors = pd.DataFrame(columns=['frameNum', 'raftID', 'hexaticOrderParameter',
                                        'neighborDistances', 'neighborDistancesAvg'])

    dfNeighborsAllFrames = pd.DataFrame(columns=['frameNum', 'raftID', 'hexaticOrderParameter',
                                                 'neighborDistances', 'neighborDistancesAvg'])

    currStepNum = 0   
  
    


    if initialPositionMethod == 1:
        #这个位置再次提到了初始化方法，以哪次为准呢？
        # initialize the raft positions in the first frame, check pairwise ccdistance all above 2R
        paddingAroundArena = 5  # unit: radius
        ccDistanceMin = 2.5  # unit: radius
        raftLocations[:, currStepNum, :] = np.random.uniform(0 + raftRadius * paddingAroundArena,
                                                             arenaSize - raftRadius * paddingAroundArena,
                                                             (numOfRafts, 2))
        raftsToRelocate = np.arange(numOfRafts)
        while len(raftsToRelocate) > 0:
            raftLocations[raftsToRelocate, currStepNum, :] = np.random.uniform(
                0 + raftRadius * paddingAroundArena,
                arenaSize - raftRadius * paddingAroundArena, (len(raftsToRelocate), 2))
            pairwiseDistances = scipy_distance.cdist(raftLocations[:, currStepNum, :],
                                                     raftLocations[:, currStepNum, :], 'euclidean')
            np.fill_diagonal(pairwiseDistances, raftRadius * ccDistanceMin + 1)
            raftsToRelocate, _ = np.nonzero(pairwiseDistances < raftRadius * ccDistanceMin)
            raftsToRelocate = np.unique(raftsToRelocate)
    elif initialPositionMethod == 2 or (initialPositionMethod == 3 and firstSpinSpeedFlag == 1):
        if numOfRafts == 2:
            raftLocations[0, currStepNum, :] = np.array([arenaSize / 2 + ccSeparationStarting / 2, arenaSize / 2])
            raftLocations[1, currStepNum, :] = np.array([arenaSize / 2 - ccSeparationStarting / 2, arenaSize / 2])
            raftOrientations[:, currStepNum] = initialOrientation
            raftRotationSpeedsInRad[:, currStepNum] = omegaBField
        else:
            raftLocations[:, currStepNum, :] = fsr.square_spiral(numOfRafts, raftRadius * 2 + 100, centerOfArena)
        firstSpinSpeedFlag = 0
    elif initialPositionMethod == 3 and firstSpinSpeedFlag == 0:
        raftLocations[0, currStepNum, :] = lastPositionOfPreviousSpinSpeeds[0, :]
        raftLocations[1, currStepNum, :] = lastPositionOfPreviousSpinSpeeds[1, :]



    '''相同名称的视频会自动覆盖，碰到参数合适的要记下来，要不就找不到了😭'''
    outputFileName = 'Simulation_' + solverMethod + '_' + str(numOfRafts) + 'Rafts_' \
                     + str(magneticFieldRotationRPS).zfill(3) + 'rps_B' + str(magneticFieldStrength) \
                     + 'T_m' + str(magneticMomentOfOneRaft) \
                     + '_startPosMeth' + str(initialPositionMethod) \
                     + '_timeStep' + str(timeStepSize) + '_' + str(timeTotal) + 's'
 
    if outputVideo == 1:
        outputVideoName = outputFileName + '.mp4'
        fourcc = cv.VideoWriter_fourcc(*'mp4v')  # *'mp4v' worked for linux, *'DIVX', MJPG
        frameW, frameH, _ = blankFrameBGR.shape
        videoOut = cv.VideoWriter(outputVideoName, fourcc, outputFrameRate, (frameH, frameW), 1)
  
    for currStepNum in progressbar.progressbar(np.arange(0, numOfTimeSteps - 1)):
        # currentStepNum = 0
        magneticFieldDirection = (magneticFieldRotationRPS * 360 * currStepNum * timeStepSize) % 360

        raftLocationsOrientations = np.concatenate((raftLocations[:, currStepNum, :].flatten(),
                                                    raftOrientations[:, currStepNum]))
 
        sol = solve_ivp(funct_drdt_dalphadt, (0, timeStepSize), raftLocationsOrientations, method=solverMethod)

        raftLocations[:, currStepNum + 1, :] = sol.y[0:numOfRafts * 2, -1].reshape(numOfRafts, 2)
        raftOrientations[:, currStepNum + 1] = sol.y[numOfRafts * 2: numOfRafts * 3, -1]



        if numOfRafts > 2:
            # Voronoi calculation:
            vor = scipyVoronoi(raftLocations[:, currStepNum, :])
            allVertices = vor.vertices
            neighborPairs = vor.ridge_points  # row# is the index of a ridge,
            # columns are the two point# that correspond to the ridge

            ridgeVertexPairs = np.asarray(vor.ridge_vertices)  # row# is the index of a ridge,
            # columns are two vertex# of the ridge

            pairwiseDistances = scipy_distance.cdist(raftLocations[:, currStepNum, :],
                                                     raftLocations[:, currStepNum, :], 'euclidean')

 
            # calculate hexatic order parameter and entropy by neighbor distances
            for raftID in np.arange(numOfRafts):
                # raftID = 0
                r_i = raftLocations[raftID, currStepNum, :]  # unit: micron

                # neighbors of this particular raft:
                ridgeIndices0 = np.nonzero(neighborPairs[:, 0] == raftID)
                ridgeIndices1 = np.nonzero(neighborPairs[:, 1] == raftID)
                ridgeIndices = np.concatenate((ridgeIndices0, ridgeIndices1), axis=None)
                neighborPairsOfOneRaft = neighborPairs[ridgeIndices, :]
                NNsOfOneRaft = np.concatenate((neighborPairsOfOneRaft[neighborPairsOfOneRaft[:, 0] == raftID, 1],
                                               neighborPairsOfOneRaft[neighborPairsOfOneRaft[:, 1] == raftID, 0]))
                neighborDistances = pairwiseDistances[raftID, NNsOfOneRaft]

                # calculate hexatic order parameter of this one raft
                neighborLocations = raftLocations[NNsOfOneRaft, currStepNum, :]
                neighborAnglesInRad = np.arctan2(-(neighborLocations[:, 1] - r_i[1]),
                                                 (neighborLocations[:, 0] - r_i[0]))
                # negative sign to make angle in the right-handed coordinate

                raftHexaticOrderParameter = \
                    np.cos(neighborAnglesInRad * 6).mean() + np.sin(neighborAnglesInRad * 6).mean() * 1j

                dfNeighbors.at[raftID, 'frameNum'] = currStepNum
                dfNeighbors.at[raftID, 'raftID'] = raftID
                dfNeighbors.at[raftID, 'hexaticOrderParameter'] = raftHexaticOrderParameter
                dfNeighbors.at[raftID, 'neighborDistances'] = neighborDistances
                dfNeighbors.at[raftID, 'neighborDistancesAvg'] = neighborDistances.mean()


            # calculate order parameters for the current time step:
            hexaticOrderParameterList = dfNeighbors['hexaticOrderParameter'].tolist()
            neighborDistancesList = np.concatenate(dfNeighbors['neighborDistances'].tolist())

            hexaticOrderParameterArray = np.array(hexaticOrderParameterList)
            '''
            hexaticOrderParameterAvgs[currStepNum] = hexaticOrderParameterArray.mean()
            hexaticOrderParameterAvgNorms[currStepNum] = np.sqrt(hexaticOrderParameterAvgs[currStepNum].real ** 2
                                                                 + hexaticOrderParameterAvgs[currStepNum].imag ** 2)
            '''
            '''
            hexaticOrderParameterMeanSquaredDeviations[currStepNum] = ((hexaticOrderParameterArray
                                                                        - hexaticOrderParameterAvgs[
                                                                            currStepNum]) ** 2).mean()
            '''
            hexaticOrderParameterModulii = np.absolute(hexaticOrderParameterArray)
            hexaticOrderParameterModuliiAvgs[currStepNum] = hexaticOrderParameterModulii.mean()
            '''
            hexaticOrderParameterModuliiStds[currStepNum] = hexaticOrderParameterModulii.std()
            '''
            count, _ = np.histogram(np.asarray(neighborDistancesList) / raftRadius, binEdgesNeighborDistances)
            entropyByNeighborDistances[currStepNum] = fsr.shannon_entropy(count)

            '''
            # g(r) and g6(r) for this frame
            for radialIndex, radialIntervalStart in enumerate(radialRangeArray):
                radialIntervalEnd = radialIntervalStart + deltaR
                # g(r)
                js, ks = np.logical_and(pairwiseDistances >= radialIntervalStart,
                                        pairwiseDistances < radialIntervalEnd).nonzero()
                count = len(js)
                density = numOfRafts / arenaSize ** 2
                radialDistributionFunction[currStepNum, radialIndex] = \
                    count / (2 * np.pi * radialIntervalStart * deltaR * density * ( numOfRafts - 1))
                # g6(r)
                sumOfProductsOfPsi6 = \
                    (hexaticOrderParameterArray[js] * np.conjugate(hexaticOrderParameterArray[ks])).sum().real
                spatialCorrHexaOrderPara[currStepNum, radialIndex] = \
                    sumOfProductsOfPsi6 / (2 * np.pi * radialIntervalStart * deltaR * density * (numOfRafts - 1))
                # g6(r)/g(r)
                if radialDistributionFunction[currStepNum, radialIndex] != 0:
                    spatialCorrHexaBondOrientationOrder[currStepNum, radialIndex] = \
                        spatialCorrHexaOrderPara[currStepNum, radialIndex] / radialDistributionFunction[
                            currStepNum, radialIndex]
            '''
            # dfNeighborsAllFrames = dfNeighborsAllFrames.append(dfNeighbors,ignore_index=True)       
        if (outputImageSeq == 1 or outputVideo == 1) and (currStepNum % intervalBetweenFrames == 0):
            currentFrameBGR = fsr.draw_rafts_rh_coord(blankFrameBGR.copy(),
                                                      np.int32(raftLocations[:, currStepNum, :] / scaleBar),
                                                      np.int64(raftRadii / scaleBar), numOfRafts)

            
        if numOfRafts == 2:
                currentFrameBGR = fsr.draw_b_field_in_rh_coord(currentFrameBGR, magneticFieldDirection)
                currentFrameBGR = fsr.draw_raft_orientations_rh_coord(currentFrameBGR,
                                                                  np.int64(
                                                                      raftLocations[:, currStepNum, :] / scaleBar),
                                                                  raftOrientations[:, currStepNum],
                                                                  np.int64(raftRadii / scaleBar), numOfRafts)
                currentFrameBGR = fsr.draw_raft_num_rh_coord(currentFrameBGR,
                                                         np.int64(raftLocations[:, currStepNum, :] / scaleBar),
                                                         numOfRafts)     
                
        if numOfRafts == 2:
                vector1To2SingleFrame = raftLocations[1, currStepNum, :] - raftLocations[0, currStepNum, :]
                distanceSingleFrame = np.sqrt(vector1To2SingleFrame[0] ** 2 + vector1To2SingleFrame[1] ** 2)
                phase1To2SingleFrame = np.arctan2(vector1To2SingleFrame[1], vector1To2SingleFrame[0]) * 180 / np.pi
                currentFrameBGR = fsr.draw_frame_info(currentFrameBGR, currStepNum, distanceSingleFrame,
                                                      raftOrientations[0, currStepNum], magneticFieldDirection,
                                                      raftRelativeOrientationInDeg[0, 1, currStepNum])     
                
        elif numOfRafts > 2:
                currentFrameBGR = fsr.draw_frame_info_many(currentFrameBGR, currStepNum,
                                                           hexaticOrderParameterAvgNorms[currStepNum],
                                                           hexaticOrderParameterModuliiAvgs[currStepNum],
                                                           entropyByNeighborDistances[currStepNum])
                                                          

        if outputImageSeq == 1:
                outputImageName = outputFileName + str(currStepNum).zfill(7) + '.jpg'
                cv.imwrite(outputImageName, currentFrameBGR)
        if outputVideo == 1:
                videoOut.write(np.uint8(currentFrameBGR))
                
                
    if outputVideo == 1:
        videoOut.release()


    # %% 将关键数据储存为HDF5文件
    hdf5_file = h5py.File(outputFileName+'.h5', 'w')
    for key in listOfVariablesToSave:
        try:
            data_to_save = globals()[key]
        
            # 检查数据类型，如果是标量，则不使用压缩选项
            if isinstance(data_to_save, (int, float, str)):
                hdf5_file.create_dataset(key, data=data_to_save)
            else:
                hdf5_file.create_dataset(key, data=data_to_save, compression='gzip', compression_opts=9)
        
        except TypeError as e:
            print(f'Error occurred: {e}')
    
    print('All saved successfully.')
    hdf5_file.close()

    # %%打开、读取HDF5文件，然后画图

    hdf5_file = h5py.File(outputFileName + '.h5', 'r')

    entropy_data = hdf5_file['entropyByNeighborDistances'][:]
    hdf5_file.close()



    #每一个熵都是由current step的循环算出来的
    time_steps = np.arange(0, numOfTimeSteps - 1)

    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, entropy_data, marker='o', linestyle='-', color='b', label='Entropy by Neighbor Distances')
    plt.xlabel('Time Steps')
    plt.ylabel('Entropy')
    plt.title('Entropy by Neighbor Distances over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()









    





    
   
   
    