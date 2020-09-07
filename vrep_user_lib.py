import matplotlib.pyplot as plt
import numpy as np
import vrep as v
import math as m
##import scipy.io
import pygame
import sys



# =============================================================================
# INITIALIZATION
# =============================================================================


def initialize_Vrep_connection(sync, port):
    
    # close all previous remote connections (if any)
    v.simxFinish(-1)
    
    # enable remote API Python-side, starts a communication thread
    clientID = v.simxStart('127.0.0.1',port,True,True,5000,5)
    if clientID!=-1:
        print ('Connected to remote API server')
        
        # get connection ID to track whether connection is maintained
        connection_ID = v.simxGetConnectionId(clientID)
            
        # let V-Rep know Python has connected
        v.simxAddStatusbarMessage(clientID,'Python client connected successfully',0)
        # Set operation mode of remote API server to synchronous
        v.simxSynchronous(clientID,sync)
        return (connection_ID, clientID)
    
    else: 
        v.simxFinish(-1)
        sys.exit('Warning! Failed to connect to remote API server')
    
    

def get_timestep(clientID):
    # initialize inputBuffer for CallScriptFunction
    inputBuffer = bytearray()
    # get time step from custom function in V-Rep
    rt_code, intData, timestep ,stringData, rt_buffer = \
    v.simxCallScriptFunction(clientID,'Base',1,'get_timestep', \
    [],[],[],inputBuffer,v.simx_opmode_blocking)     
    if rt_code == 0:
        return timestep[0]
    else:
        sys.exit('get_timestep function failed to execute properly')



def initialize_single_device(clientID):
        
    # get names and handles for all joints from V-Rep
    rt_code, handles, intData, floatData, names = \
        v.simxGetObjectGroupData(clientID,v.sim_object_joint_type,0,v.simx_opmode_blocking)
    print(handles)    
    # create Joint objects and add to list
    joints = []
    for j,name in enumerate(names):
        joint = Joint(name, handles[j])
        joints.append(joint)
        
    # create Robot object with list of joints
    device = Robot(joints)
    
    # create target object
    rt_code, target_handle = v.simxGetObjectHandle(clientID,'Target',v.simx_opmode_blocking)
    target = SceneObject('Target',target_handle,None)

    # create key object
    rt_code, target_handle = v.simxGetObjectHandle(clientID,'Key',v.simx_opmode_blocking)
    key = SceneObject('Key',target_handle,None)

    # create tip of car object
    rt_code, tip_handle = v.simxGetObjectHandle(clientID,'Car',v.simx_opmode_blocking)
    car = SceneObject('Car',tip_handle,None)
      
    return (device,target,key,car)



def initialize_dual_devices(clientID):
    
    # get names and handles for all joints from V-Rep
    rt_code, handles, intData, floatData, names = \
    v.simxGetObjectGroupData(clientID,v.sim_object_joint_type,0,v.simx_opmode_blocking)
    
    # create Joint objects and add to list for both devices
    joints_left = []
    joints_right = []
    for j,name in enumerate(names):
        joint = Joint(name, handles[j])
        if '#0' in name:
            joints_left.append(joint)
        else:
            joints_right.append(joint)
            
    # create Robot objects with list of joints
    left_device = Robot(joints_left)
    right_device = Robot(joints_right)   
    
    # create target objects
    rt_code, left_target_handle = v.simxGetObjectHandle(clientID,'target#0',v.simx_opmode_blocking)
    left_target = SceneObject('target#0',left_target_handle,None)
    rt_code, right_target_handle = v.simxGetObjectHandle(clientID,'target',v.simx_opmode_blocking)
    right_target = SceneObject('target',right_target_handle,None)
    # create tip objects
    rt_code, left_tip_handle = v.simxGetObjectHandle(clientID,'tip#0',v.simx_opmode_blocking)
    left_tip = SceneObject('tip#0',left_tip_handle,None)
    rt_code, right_tip_handle = v.simxGetObjectHandle(clientID,'tip',v.simx_opmode_blocking)
    right_tip = SceneObject('tip',right_tip_handle,None)
      
    return (left_device,right_device,left_tip,right_tip,left_target,right_target)


def initialize_dummy_obstacle(clientID):
    rt_code, handles, intData, floatData, names = \
        v.simxGetObjectGroupData(clientID,v.sim_object_shape_type,0,v.simx_opmode_blocking)
    obstacles = []
    for i,name in enumerate(names):
        if 'Disc' in name:
            ob = SceneObject(name,handles[i],None)
            obstacles.append(ob)
    return obstacles


def initialize_force_sensor(clientID):
    rt_code, sensor_handles, sensor_intData, sensor_floatData, sensor_names = \
        v.simxGetObjectGroupData(clientID,v.sim_object_forcesensor_type,0,v.simx_opmode_blocking)
    sensors = []
    for i,name in enumerate(sensor_names):
        if 'force_sensor' in name.lower():
            sensor = SensorObject(name,sensor_handles[i],None)
            v.simxReadForceSensor(clientID,sensor_handles[i],v.simx_opmode_blocking)
            v.simxReadForceSensor(clientID,sensor_handles[i],v.simx_opmode_streaming)
            sensors.append(sensor)
    return sensors

def initialize_distance_stream(clientID,distance):
    rt_code, distance.handle = \
        v.simxGetDistanceHandle(clientID,distance.name,v.simx_opmode_blocking)
    rt_code, _ = \
        v.simxReadDistance(clientID,distance.handle,v.simx_opmode_blocking)
    rt_code, _ = \
        v.simxReadDistance(clientID,distance.handle,v.simx_opmode_streaming)
    return



def initialize_dummy_position_stream(clientID,dummy):
    rt_code, dummy.position = \
        v.simxGetObjectPosition(clientID,dummy.handle,-1,v.simx_opmode_blocking)
    rt_code, _ = \
        v.simxGetObjectPosition(clientID,dummy.handle,-1,v.simx_opmode_streaming)
    return


def initialize_dummy_orientation_stream(clientID,dummy):
    rt_code, dummy.orientation = \
        v.simxGetObjectOrientation(clientID,dummy.handle,-1,v.simx_opmode_blocking)
    rt_code, _ = \
        v.simxGetObjectOrientation(clientID,dummy.handle,-1,v.simx_opmode_streaming)
    return


def initialize_device_streams(clientID,device):
    # initialize position, velocity and force/torque streams
    for n,joint in enumerate(device.joints):
        # joint positions
        rt_code, device.joints[n].position = \
            v.simxGetJointPosition(clientID,joint.handle,v.simx_opmode_streaming)
        # joint forces/torques
        rt_code, device.joints[n].force = \
            v.simxGetJointForce(clientID,joint.handle,v.simx_opmode_streaming)
        # joint velocities
        v.simxSetJointTargetVelocity(clientID,joint.handle,0,v.simx_opmode_streaming)
    return

def initialize_collision_stream(clientID,collision):
    rt_code, collision.handle = \
             v.simxGetCollisionHandle(clientID,collision.name,v.simx_opmode_blocking)
    rt_code, _ = \
             v.simxReadCollision(clientID, collision.handle,v.simx_opmode_blocking)
    rt_code, _ = \
             v.simxReadCollision(clientID, collision.handle,v.simx_opmode_streaming)    



def initialize_joystick():    
    # scan PC for joysticks and initialize
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 1:
        print('Joystick connected.')
        Joystick = pygame.joystick.Joystick(0)    
        Joystick.init()
    elif pygame.joystick.get_count() == 0:
        sys.exit('Could not find a joystick.')
    else:
        print('Multiple joysticks connected, might not work as intended.')
        Joystick = pygame.joystick.Joystick(0)    
        Joystick.init()
    return Joystick


# =============================================================================
# MODELING
# =============================================================================


# Joint has a name, handle, position and force 
class Joint:
    def __init__ (self, name, handle):
        self.name = name
        self.handle = handle
        self.position = None
        self.velocity = None
        self.error = None
        self.force = None
        return        



# Robot consists of a list of joints
class Robot:
    def __init__(self,joints):
        self.joints = joints
        self.n_joints = len(joints)
        return
    
    
    
class SceneObject:
    def __init__(self, name, handle, position, orientation=None):
        self.name = name
        self.handle = handle
        self.position = position
        self.orientation = orientation
        return

class SensorObject:
    def __init__(self, name, handle, position):
        self.name = name
        self.handle = handle
        self.position = position
        return
    
class Distance:
    def __init__(self, name, handle, value):
        self.name = name
        self.handle = handle
        self.value = value
        return

class Collision:
    def __init__(self, name, handle=None, state=None):
        self.name = name
        self.handle = handle
        self.state = None


def apply_friction(position_old,position_new,timestep,input_force,f_static,
                   f_viscous,f_kinetic):
    # Use two positions to calculate velocity
    velocity = (position_new - position_old)/timestep
    v_limit = f_static/f_viscous
    if abs(velocity) <= v_limit:
        applied_force = input_force - f_viscous * velocity
    else:
        applied_force = input_force - f_kinetic * np.sign(velocity)
    return (applied_force)
    

# =============================================================================
# REFERENCE SIGNALS
# =============================================================================

     
def minimum_jerk(x_start,x_end,time):
    # generates a straight minimum-jerk movement between two points 
    duration = time[-1] - time[0]
    x =  x_start+(x_end-x_start)*(10*pow(time/duration,3)- \
        15*pow(time/duration,4)+6*pow(time/duration,5))
    return x



def generate_PaP_expert(timestep):
    # generates a minimum-jerk "expert" trajectory in end-effector coordinates
    # including gripper mode (open or closed) for pick & place task
    
    movements = {} # duration and classification of all parts of the motion    
    movements['left'] = [(2,'o_move'),(1,'c_grip'),(2,'c_move'),(1,'o_grip'),\
             (3,'o_wait')]
    movements['right'] = [(2,'o_move'),(3,'o_wait'),(1,'c_grip'),(2,'c_move'),\
             (1,'o_grip')]
    
    points = {} # all points of the motion (x,y,z,alpha,beta,gamma)
    points['left'] =  np.array(([-1.2829e-1, 2.0176e-4, 1.2716e-2, \
                                 2.2680, -1.0731e1, -5.4011e1], \
                                [-1.2500e-1, 0, 7.5000e-2, \
                                 1.9975e-1, 1.0956e-1, 1.1385e1], \
                                [2.0000e-2, 0, 7.5000e-2, \
                                 2.5642e-1, 2.0553e-1, -3.8848e1]))
    points['right'] = np.array(([-4.2456e-2, -5.2846e-4, 1.2804e-2, \
                                 3.2821, 9.7147, 3.5139e1], \
                                [-3.0000e-2, 0, 7.5000e-2, \
                                 1.9975e-1, -1.0956e-1, -3.8847e1], \
                                [2.5000e-2, 1.4901e-8, 7.5000e-2, \
                                 2.5642e1, 2.0553, 4.1153e1]))
    
    trajectory={}
    trajectory['left'] = np.append(points['left'][0],0).reshape((1,7))
    trajectory['right'] = np.append(points['right'][0],0).reshape((1,7))
    for side in ['left','right']:
        move = 0                        # counter for number of movements
        for step in movements[side]:    
            n = int(step[0]/timestep)   
            t = np.arange(n)*timestep   # time vector for part of the motion
            traj = np.zeros((n,7))
            if 'move' in step[1]:       # if moving
                for i in range(6):
                    traj[:,i] = minimum_jerk(points[side][move][i],points[side][move+1][i],t)
                move =+ 1               
            else:                       # if waiting or gripping
                for i in range(6):      
                    traj[:,i] = np.full(n,trajectory[side][-1,i]) # maintain position
            if 'c_' in step[1]:     # close gripper
                traj[:,6] = np.ones(n)
            else:                   # open gripper
                traj[:,6] = np.zeros(n)
            trajectory[side] = np.append(trajectory[side],traj,axis=0) # add part to trajectory
        trajectory[side] = np.delete(trajectory[side],0,0) # remove duplicate first step
    return trajectory


def generate_ref_expert(trajectory_number,side):
    expert_data = scipy.io.loadmat('expert_data.mat')
    trajectory_number
    reference_trajectory = np.column_stack((
            np.array([x for x in expert_data[side][trajectory_number][0][:,6]/1000]),
            np.array([y for y in expert_data[side][trajectory_number][0][:,7]/1000]),
            np.array([z for z in expert_data[side][trajectory_number][0][:,8]/1000])))
    time = (expert_data[side][trajectory_number][0][:,0])/1000
    timestep = np.array(np.diff(time)) # timestep differs each time
    timestep = np.append(timestep, 0)
    length = len(reference_trajectory)
    return(reference_trajectory,time,timestep,length)
    
    
    
def generate_ref_test(length,timestep):
     # time vector is length [s] divided by timestep [s]
    time = np.linspace(0,length,int(length/timestep))
    dt = np.zeros(int(length/timestep))
    dt.fill(timestep) #every timestep has the same length
    reference_trajectory = np.zeros((int(length/timestep),3))
    reference_trajectory[:,0] = np.sin(time*2*m.pi/length)/10
    reference_trajectory[:,1] = -np.sin(time*2*m.pi/length)/10
    reference_trajectory[:,2] = (-np.cos(time*4*m.pi/length)+1)/20
    return(reference_trajectory,time,dt,len(time))
    
    
    
def generate_noise(amplitude,length):
    time = np.arange(length)
    timestep = np.zeros(length)
    timestep.fill(1)
    reference_trajectory = np.random.rand(length)*amplitude
    return(reference_trajectory,time,timestep,length)


# =============================================================================
# CONTROL
# =============================================================================
    
    
def update_dummy_position(clientID,dummy):
    for _ in range(5):
        rt_code, dummy.position = \
            v.simxGetObjectPosition(clientID,dummy.handle,-1,v.simx_opmode_buffer)
        # return orienation in radian
        rt_code, dummy.orientation = \
            v.simxGetObjectOrientation(clientID,dummy.handle,-1,v.simx_opmode_buffer)


def update_distance(clientID,distance):
    for _ in range(5):
        rt_code, distance.value = \
            v.simxReadDistance(clientID,distance.handle,v.simx_opmode_buffer)


def update_device(clientID,device):
    for _ in range(5):
        for n,joint in enumerate(device.joints):
            # joint positions
            rt_code, device.joints[n].position = \
                v.simxGetJointPosition(clientID,joint.handle,v.simx_opmode_buffer)
            # joint forces/torques
            rt_code, device.joints[n].force = \
                v.simxGetJointForce(clientID,joint.handle,v.simx_opmode_buffer)
    return


def update_collision(clientID,collision):
    for _ in range(5):
        _, collision.state = v.simxReadCollision(clientID, collision.handle, v.simx_opmode_buffer)
    

def get_IK_single(clientID):
    # initialize inputBuffer for CallScriptFunction
    inputBuffer = bytearray()
    # solve IK in V-Rep and obtain desired joint positions
    rt_code, IK_handles, IK_matrix, stringData, rt_buffer = \
        v.simxCallScriptFunction(clientID,'Base',1,'get_IK', \
        [],[],[],inputBuffer,v.simx_opmode_blocking) 
    return (IK_handles, IK_matrix)



def get_IK_dual(clientID):
    # initialize inputBuffer for CallScriptFunction
    inputBuffer = bytearray()
    # solve IK in V-Rep and obtain desired joint positions
    rt_code, IK_handles_left, IK_matrix_left, stringData, rt_buffer = \
        v.simxCallScriptFunction(clientID,'Base#0',1,'get_IK_left', \
        [],[],[],inputBuffer,v.simx_opmode_blocking) 
    rt_code, IK_handles_right, IK_matrix_right, stringData, rt_buffer = \
        v.simxCallScriptFunction(clientID,'Base',1,'get_IK_right', \
        [],[],[],inputBuffer,v.simx_opmode_blocking) 
    return (IK_handles_left,IK_matrix_left,IK_handles_right,IK_matrix_right)



class DOF_input:
    def __init__(self):
        self.value = None
        self.handle = None
        self.force = None
        return
    
    
   
class User_input:
    def __init__(self):
        self.inputs = {'yaw':DOF_input(),
                       'pitch':DOF_input(),
                       'roll':DOF_input(),
                       'insertion':DOF_input(),
                       'gripper':DOF_input()}
        return
    
    def associate_handles(self,yprig_handles):
        for i,DOF in enumerate(self.inputs):
            self.inputs[DOF].handle = yprig_handles[i]
        return
    
    def associate_forces(self,yprig_forces):
        for i,DOF in enumerate(self.inputs):
            self.inputs[DOF].force = yprig_forces[i]
        return
    
    def associate_friction(self,f_static,f_viscous,f_kinetic):
        for i,DOF in enumerate(self.inputs):
            self.inputs[DOF].friction = (f_static[i],f_viscous[i],f_kinetic[i])
        return

    def get_joystick_input_left(self,Joystick):
        # Uses Logitech Gamepad F310
        # check for events
        pygame.event.get()
        # get joystick states and update user inputs
        self.inputs['yaw'].value = Joystick.get_axis(1)
        self.inputs['pitch'].value = -Joystick.get_axis(0)
        self.inputs['roll'].value = -Joystick.get_hat(0)[0]
        self.inputs['insertion'].value = -Joystick.get_hat(0)[1]
        self.inputs['gripper'].value = -np.clip(Joystick.get_axis(2),0,1) + 0.5
        return
     
    def get_joystick_input_right(self,Joystick):
        # Logitech Gamepad F310
        pygame.event.get()
        self.inputs['yaw'].value = -Joystick.get_axis(3)
        self.inputs['pitch'].value = Joystick.get_axis(4)
        if Joystick.get_button(1) == 1 and Joystick.get_button(2) == 0:
            self.inputs['roll'].value = -1
        elif Joystick.get_button(1) == 0 and Joystick.get_button(2) == 1:
            self.inputs['roll'].value = 1
        else:
            self.inputs['roll'].value = 0
        if Joystick.get_button(0) == 1 and Joystick.get_button(1) == 0:
            self.inputs['insertion'].value = 1
        elif Joystick.get_button(0) == 0 and Joystick.get_button(1) == 1:
            self.inputs['insertion'].value = -1
        else:
            self.inputs['insertion'].value = 0
        self.inputs['gripper'].value = np.clip(Joystick.get_axis(2),-1,0) + 0.5
        return  
    


def PD_control(K_p,K_d,error,previous_error,timestep,u_max):
        de = (error - previous_error)/timestep
        u = np.clip(K_p*error + K_d*de,-u_max,u_max)
        return(u)
         
        
        
def apply_force(clientID,force,handle):
    if force > 0:
        v.simxSetJointForce(clientID,handle,force,v.simx_opmode_oneshot)
        v.simxSetJointTargetVelocity(clientID,handle,1000000,v.simx_opmode_oneshot)
    else:
        v.simxSetJointForce(clientID,handle,-force,v.simx_opmode_oneshot)
        v.simxSetJointTargetVelocity(clientID,handle,-1000000,v.simx_opmode_oneshot)
    return
    

# =============================================================================
# PLOTTING 
# =============================================================================
        
    
def plot_reference_xyz(time,reference_trajectory,tip_position,target_position,\
                       error):
    fig,ax = plt.subplots(2,1)
    ax[0].plot(time,target_position[:,0], label = 'x target V-Rep')
    ax[0].plot(time,target_position[:,1], label = 'y target V-Rep')
    ax[0].plot(time,target_position[:,2], label = 'z target V-Rep')
    ax[0].plot(time,tip_position[:,0], label = 'x tip V-Rep')
    ax[0].plot(time,tip_position[:,1], label = 'y tip V-Rep')
    ax[0].plot(time,tip_position[:,2], label = 'z tip V-Rep')    
    ax[0].legend(loc="lower left")
    ax[0].set_xlabel('time [s]')
    ax[0].set_ylabel('position [m]')
        
    ax[1].plot(time,error[:,0], label = 'x error')
    ax[1].plot(time,error[:,1], label = 'y error')
    ax[1].plot(time,error[:,2], label = 'z error')
    ax[1].legend(loc="lower left")
    ax[1].set_xlabel('time [s]')
    ax[1].set_ylabel('positioning error [m]')
    return
    


def plot_joint_tracking_results(time,joint_error,forces,joint_position,IK_results,\
                       tip_position,target_position,device):
    
    joint_columns = {}
    for j,joint in enumerate(device.joints):
        if joint.name == 'Rotation_1_active':
            joint_columns['pitch'] = (0,j)
        elif joint.name == 'Rotation_2_passive':
            joint_columns['yaw'] = (1,j)
        elif joint.name == 'Rotation_5_active_shaft':
            joint_columns['roll'] = (2,j)
        elif joint.name == 'Translation_1_passive':
            joint_columns['insertion'] = (3,j)

    for i in ['yaw', 'pitch', 'roll', 'insertion']:
         
        fig, ax = plt.subplots(2,2)
        ax[0,0].plot(time,joint_position[:,joint_columns[i][1]], label = 'measured')
        ax[0,0].plot(time,IK_results[:,joint_columns[i][0]], label = 'desired')
        ax[0,0].legend(loc="upper left")
        ax[0,0].set_xlabel('time [s]'); ax[0,0].set_ylabel('joint position [m, rad]')
        ax[0,1].plot(time,joint_error[:,joint_columns[i][0]])
        ax[0,1].set_xlabel('time [s]'); ax[0,1].set_ylabel('error [m, rad]')
        ax[1,0].plot(time,forces[:,joint_columns[i][1]])
        ax[1,0].set_xlabel('time [s]'); ax[1,0].set_ylabel('force applied to joint [N, Nm]')
        
        ax[1,1].plot(time,target_position[:,0], label = 'x reference')
        ax[1,1].plot(time,target_position[:,1], label = 'y reference')
        ax[1,1].plot(time,target_position[:,2], label = 'z reference')
        ax[1,1].plot(time,tip_position[:,0], label = 'x tip')
        ax[1,1].plot(time,tip_position[:,1], label = 'y tip')
        ax[1,1].plot(time,tip_position[:,2], label = 'z tip')
        ax[1,1].legend(loc="lower left")
        ax[1,1].set_xlabel('time [s]'); ax[1,1].set_ylabel('absolute position [m]')    
        fig.suptitle('simulation results '+i)
        plt.show()
        
        plt.show() 
    return



def plot_joint_joystick_results(positions,forces):
    timesteps = np.arange(len(positions))
    
    for i,DOF in zip([0,1,2,3,4],['yaw','pitch','roll','insertion','gripper']):
        fig1,ax = plt.subplots(2,1)
        ax[0].plot(timesteps,forces['input'][:,i], label = 'input')
        ax[0].plot(timesteps,forces['net'][:,i], label = 'net')
        ax[0].plot(timesteps,forces['applied'][:,i], label = 'applied')
        ax[0].legend(loc="lower left")
        ax[0].set_xlabel('time step [-]')
        ax[0].set_ylabel('force [N,Nm]')
        ax[1].plot(timesteps,positions[:,i],label = 'position')
        ax[1].legend(loc="lower left")
        ax[1].set_xlabel('time step [-]')
        ax[1].set_ylabel('position [m,rad]')
        fig1.suptitle('Input Forces and Position '+DOF)
#        fig2 = plt.figure()
#        plt.plot(timesteps,velocities[:,i])
#        fig2.suptitle('Velocity'+DOF)
    return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

