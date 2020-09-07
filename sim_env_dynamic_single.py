import vrep
import time
import numpy as np
import vrep_user_lib as vu
import vrep as v
import random
from math import e

class ppo_Car_env:
    def __init__(self, timestep, port, fast=False, time_episode = 0.1):
        
        # intializing variables
        self.floor = np.asarray([[-0.95, -0.95],[0.95, 0.95]]) # upper left corner, lower right corner
        self.timestep = timestep
        self.time_episode = time_episode
        self.total_time = 0
        self.done = 0
        self.reward = 0
        self.collision_soft = 0
        self.collision_hard = 0
        self.threshold_soft = 0.01
        self.threshold_hard = 0.15
        self.threshold_crit = 0.25
        self.got_key = 0
        self.port = port
        self.fast = fast
        
        #Initializing V-Rep connection and obtaining simulation time step
        _, self.clientID = vu.initialize_Vrep_connection(True, self.port)
##        try:
##            status = v.simxLoadScene(self.clientID, 'ppo_car.ttt', 1, v.simx_opmode_blocking)
##        except:
##            print('Failed to load scene')
##            print('Error code: ',status)

        self._init_handles()
        

    def _init_handles(self):
        # intializing device and dummmy structures
        self.device, self.target, self.key, self.car = vu.initialize_single_device(self.clientID)
        self.obstacles = vu.initialize_dummy_obstacle(self.clientID)
        
        # initializing distance structure and stream
        self.reward_distance = vu.Distance('dist_to_target',None,None)
        self.key_distance = vu.Distance('dist_to_key',None,None)
        self.collision_distance = vu.Distance('dist_to_collision',None,None)
        self.dist_list = [self.reward_distance, self.key_distance,self.collision_distance]
        for dist in self.dist_list:
            vu.initialize_distance_stream(self.clientID, dist)
        
        # initializing position and force streams
        vu.initialize_dummy_position_stream(self.clientID,self.key)
        vu.initialize_dummy_position_stream(self.clientID,self.target)
        vu.initialize_dummy_position_stream(self.clientID,self.car)
        
        for ob in self.obstacles:
            vu.initialize_dummy_position_stream(self.clientID,ob)
            
        
        vu.initialize_device_streams(self.clientID,self.device)

        # initializing force sensor
        self.force_sensors = vu.initialize_force_sensor(self.clientID)

        # car orientation
        vu.initialize_dummy_orientation_stream(self.clientID,self.car)

        # Collision detection
        self.collision = vu.Collision('Collision')
##        vu.initialize_collision_stream(self.clientID, self.collision)


    def action_update(self, action, dummy=False):
        
        self.collision_soft = 0
        self.collision_hard = 0
        self.collision.state = 0

##        action = np.clip(action, -2, 2)
        self.current_v = [action[0]*5, action[1]*5]
##        if not dummy:
##            v_ratio = abs(action[0]/action[1])
##            self.current_v = np.array([np.sign(action[0])*v_ratio,\
##                                       np.sign(action[1])*1/v_ratio])
        
        for i,joint in enumerate(self.device.joints):
            v.simxSetJointTargetVelocity(self.clientID,joint.handle,action[i]*5,v.simx_opmode_oneshot)
        
        # execute simulation step in V-Rep and wait for it to finish
        vrep.simxSynchronousTrigger(self.clientID)
        vrep.simxGetPingTime(self.clientID)

        # get positions, forces
        vu.update_device(self.clientID,self.device)
        for ob in self.obstacles:
            vu.update_dummy_position(self.clientID,ob)
        vu.update_dummy_position(self.clientID,self.car)
        for dist in self.dist_list:
            vu.update_distance(self.clientID,dist)
##        vu.update_distance(self.clientID,self.reward_distance)
##        vu.update_distance(self.clientID,self.key_distance)
##        vu.update_collision(self.clientID,self.collision)
        
        if not dummy:
            if abs(self.car.position[0]) >= 1.25 or abs(self.car.position[1]) >= 1.25:
                print("Out of bound")
                self.done = 1

            if self.car.position[-1] >= 0.058:
                print('Flipped over')
                self.done = 1

            if self.collision_distance.value <= 0.008:
                self.collision.state = 1
            
        # read force sensor
            for sensor in self.force_sensors:
        ##            print(v.simxReadForceSensor(self.clientID, sensor.handle,v.simx_opmode_buffer))
                _,_,f,_ = v.simxReadForceSensor(self.clientID, sensor.handle,v.simx_opmode_buffer)
                f = np.asarray(f[:2])
                if self.collision.state:                
                    self.collision_soft += 1
                    if any(f >= self.threshold_hard) or any(f <= -self.threshold_hard):
                        self.collision_hard += 1
                        self.collision_soft -= 1
                    elif any(f > self.threshold_crit) or any(f < -self.threshold_crit):
                            print("Critical collision")
                            self.done = 1
                            self.collision_soft = 0
                            self.collision_hard = 0
                            
        # Can try to zero joint velocity before next step (easier environment)
##        for i,joint in enumerate(self.device.joints):
##            v.simxSetJointTargetVelocity(self.clientID,joint.handle,0,v.simx_opmode_oneshot)
##        vrep.simxSynchronousTrigger(self.clientID)

    def get_rewards(self):

        self.total_time += self.timestep

        self.reward = 0  - self.collision_state * 0.5  #made up coeff
##        self.reward = 0 - self.total_time * 0.0005/self.time_episode - self.collision_soft * 0.005 - self.collision_hard * 0.01 #made up coeff

        if self.done:               # If early terminate by critical collision or out of bound  
            return -50, self.done
        
        if (self.key_distance.value <= 0.05 and self.got_key == 0):
            print('Got the key')
            self.got_key = 1
            self.reward += 80
            return self.reward*e**(-self.total_time/self.time_episode), 1        

        if (self.key_distance_old - self.key_distance.value) > 0:
##            key_rew = 5 * (1-self.got_key) * (self.key_distance_old - self.key_distance.value)
            key_rew = (1 - self.got_key) * 0.5 * (2.7 - self.key_distance.value)
            
        else:
            key_rew =(1 - self.got_key) * (-0.4)
##            key _rew = (self.key_distance_old - self.key_distance.value)
        self.key_distance_old = self.key_distance.value

        if (self.reward_distance_old - self.reward_distance.value) > 0:
            reward_rew = 5 * self.got_key * (self.reward_distance_old - self.reward_distance.value)
            self.reward_distance_old = self.reward_distance.value
        else:
            reward_rew = self.got_key * (self.reward_distance_old - self.reward_distance.value)

        self.reward += 0.1 * (key_rew + reward_rew)
        
        if self.total_time >= self.time_episode:
            self.done = 1
        
        return self.reward, self.done


    def get_states(self):
        self.collision_state = 0
        if self.collision_soft:
            self.collision_state = 1
        elif self.collision_hard:
            self.collision_state = 2
        states = [self.collision_state]    # 1
        states.extend(self.current_v)      # 3
        states.extend(self.car.position)   # 6
        states.append(self.car.orientation[-1])  #7
        if self.got_key == 0:
            if (self.key.position[0]-self.car.position[0]):
                y = (self.key.position[1]-self.car.position[1])
                x = (self.key.position[0]-self.car.position[0])
                key_direction = np.arctan2(y,x)
            else:
                key_direction = np.pi/2    
            states.append(key_direction)   # 8
            for obj in self.obstacles + [self.key]:  # 7 * 2 = 14
                states.extend(obj.position[:-1])  #  8 + 14 = 22
        else:
            if (self.target.position[0]-self.car.position[0]):
                r = (self.target.position[1]-self.car.position[1])/(self.target.position[0]-self.car.position[0])
                targ_direction = np.arctan(r)
            else:
                targ_direction = np.pi/2
            states.append(targ_direction)
            for obj in self.obstacles + [self.target]:
                states.extend(obj.position[:-1])            
        states = np.asarray(states)
        states = states.flatten()
        
        return states[np.newaxis,...]

        
    def reset_sim(self):
        # set joint forces to 0
        for joint in self.device.joints:
            joint.force = 0.0
        
        # stop and restart simulation
        self.stop_simulation()
##        time.sleep(0.1)
        vrep.simxSynchronous(self.clientID, True)
        self._set_floor_plan()
        self._init_handles()

        # execute step in new simulation
        vrep.simxSynchronousTrigger(self.clientID)
        vrep.simxGetPingTime(self.clientID)

        self.total_time = 0
        self.done = 0
        self.reward = 0
        self.collision_soft = 0
        self.collision_hard = 0
        self.got_key = 0
        self.start_simulation()
        
        
    def _set_floor_plan(self):
        _, carBody_handle = v.simxGetObjectHandle(self.clientID, 'dr12', v.simx_opmode_blocking)
        # randomize the position of obstacles and car
        pos_list = zip(random.sample(range(1,1000,25),len(self.obstacles)+1),random.sample(range(1,1000,25),len(self.obstacles)+1))
        scale = 1.9/1000           # Base on self.floor
##        for i,pos in enumerate(pos_list):
##            position = [pos[0]*scale - 0.95, pos[1]*scale - 0.95, 0.0707]
##            # First item is the car position
##            if i == 0:
##                v.simxSetObjectPosition(self.clientID, carBody_handle, -1, position, v.simx_opmode_oneshot)
##                car_pos = [pos[0]*scale - 0.95, pos[1]*scale - 0.95, 0.0707]
##            else:
##                position = [pos[0]*scale - 0.95, pos[1]*scale - 0.95, 0.1]
##                if (abs(position[0] - car_pos[0]) < 0.2) or (abs(position[1] - car_pos[1]) < 0.2):
##                    position[0] += 0.3
##                    position[1] += 0.3
##                v.simxSetObjectPosition(self.clientID,self.obstacles[i-1].handle,-1,position,\
##                                    v.simx_opmode_oneshot)

        # randomize target, key location
        car_pos = [0, 0, 0.0707]
        coordinate = {}
        got_coord = False
        while not got_coord:
            pos_list = list(zip(random.sample(range(1,1000,10),2),random.sample(range(1,1000,10),2)))
            for i,pos in enumerate(pos_list):
                coordinate[i] = np.array([pos[0]*scale - 0.95, pos[1]*scale - 0.95,0])
                got_coord = True
                if (np.linalg.norm(coordinate[i][:-1] - np.asarray(car_pos[:-1]),ord=2) < 0.3):
                    got_coord = False
                    break
        #Set key and target location
        v.simxSetObjectPosition(self.clientID,self.target.handle,-1,coordinate[0],\
                        v.simx_opmode_oneshot)
        v.simxSetObjectPosition(self.clientID,self.key.handle,-1,coordinate[1],\
                        v.simx_opmode_oneshot)

        # randomize car orientation
##        beta = np.random.uniform(-1,1,1) * 180
##        v.simxSetObjectOrientation(self.clientID, carBody_handle, -1, [90.0,beta,90],\
##                                                                         v.simx_opmode_oneshot)

    def start_simulation(self):
        vrep.simxStartSimulation(self.clientID,vrep.simx_opmode_blocking)
        if self.fast:
            vrep.simxSetBooleanParameter(self.clientID,
                                         vrep.sim_boolparam_display_enabled,
                                         False,
                                         vrep.simx_opmode_oneshot)
        self.action_update([0,0],dummy=True)
        self.reward_distance_old = self.reward_distance.value
        self.key_distance_old = self.key_distance.value
        
    def stop_simulation(self):
        vrep.simxStopSimulation(self.clientID,vrep.simx_opmode_blocking)
        
    def close_simulation(self):
        vrep.simxFinish(self.clientID)
