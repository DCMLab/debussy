from wavescapes import *
from glob import glob
import numpy as np
import math



def compute_magnitude_entropy(score, ver_ratio=0.2, hor_ratio=(0,1), aw_size=4):
    arr1 = produce_pitch_class_matrix_from_filename(score, aw_size=aw_size)
    utm = np.abs(apply_dft_to_pitch_class_matrix(arr1))
    vec = []
    for i in range(7):
        vec.append(utm[int(utm.shape[0] * ver_ratio)-1,:,i][utm[int(utm.shape[0] * ver_ratio)-1,:,i] != 0])
    sel = np.array([ve[int(utm.shape[1] * hor_ratio[0]):int(utm.shape[1] * hor_ratio[1])] for ve in vec])
    entr = ent.spectral_entropy(sel, 1, method='fft')
    return entr[1:]

def compute_magnitudes(score, ver_ratio=0.2, hor_ratio=(0,1), aw_size=4):
    arr1 = produce_pitch_class_matrix_from_filename(score, aw_size=aw_size)
    utm = np.abs(apply_dft_to_pitch_class_matrix(arr1))
    vec = []
    for i in range(7):
        vec.append(utm[int(utm.shape[0] * ver_ratio)-1,:,i][utm[int(utm.shape[0] * ver_ratio)-1,:,i] != 0])
    sel = np.array([ve[int(utm.shape[1] * hor_ratio[0]):int(utm.shape[1] * hor_ratio[1])] for ve in vec])
    return sel[4]

def compute_peaks(score, ver_ratio=0.2, hor_ratio=(0,1), aw_size=4):
    arr1 = produce_pitch_class_matrix_from_filename(score, aw_size=aw_size)
    utm = np.abs(apply_dft_to_pitch_class_matrix(arr1))
    vec = []
    for i in range(7):
        vec.append(utm[int(utm.shape[0] * ver_ratio)-1,:,i][utm[int(utm.shape[0] * ver_ratio)-1,:,i] != 0])
    sel = [ve[int(utm.shape[1] * hor_ratio[0]):int(utm.shape[1] * hor_ratio[1])] for ve in vec]
    return [len(find_peaks(list(magnitudes))[0]) for magnitudes in sel]

def compute_entropy_phase(score, ver_ratio=0.2, hor_ratio=(0,1), aw_size=4):
    arr1 = produce_pitch_class_matrix_from_filename(score, aw_size=aw_size)
    utm = np.round(np.angle(apply_dft_to_pitch_class_matrix(arr1)), 2)
    vec = []
    coeffs = []
    height = int(utm.shape[0] * ver_ratio)-1
    sel = np.array(utm[height,:,:]).T
    entr = ent.spectral_entropy(sel, 1, method='fft')
    return entr[1:]

def most_resonant(score, aw_size=4):
    arr1 = produce_pitch_class_matrix_from_filename(score, aw_size)
    utm = apply_dft_to_pitch_class_matrix(arr1)
    utm_magnitude = np.abs(utm)
    utm_max = np.max(utm_magnitude[:,:,1:], axis=2)
    utm_argmax = np.argmax(utm_magnitude[:,:,1:], axis=2)

    return (utm_max, utm_argmax, utm)


def stand(v):
    return int(v*0xff)

def two_pi_modulo(value):
    value = value*2*math.pi/6
    return np.mod(value, 2*math.pi)
    
def step_function_quarter_pi_activation(lo_bound, hi_bound, value):
    #in the increasing path branch
    if value >= lo_bound and value <= lo_bound + math.pi/3:
        return ((value-lo_bound)/(math.pi/3))
    #in the decreasing path branch
    elif value >= hi_bound and value <= hi_bound + math.pi/3:
        return 1-((value-hi_bound)/(math.pi/3))
    else:
        #the case of red 
        if lo_bound > hi_bound:
            return 0 if value > hi_bound and value < lo_bound else 1
        else:
            return 1 if value > lo_bound and value < hi_bound else 0
    
def circular_hue_revised(angle, magnitude=1.):
    # np.angle returns value in the range of [-pi : pi], where the circular hue is defined for
    # values in range [0 : 2pi]. Rather than shifting by a pi, the solution is for the negative
    # part to be mapped to the [pi: 2pi] range which can be achieved by a modulo operation.
            
    #Need to shift the value with one pi as the range of the angle given is between pi and minus pi
    #and the formulat I use goes from 0 to 2pi.
    angle = two_pi_modulo(angle)
    green = lambda a: step_function_quarter_pi_activation(0, math.pi, a)
    blue = lambda a: step_function_quarter_pi_activation(math.pi*2/3, math.pi*5/3, a)
    red = lambda a: step_function_quarter_pi_activation(math.pi*4/3, math.pi/3, a)
    value = (stand(red(angle)), stand(green(angle)), stand(blue(angle)), stand(magnitude))
    
    return value

def zeroth_coeff_cm(value, curr_max, curr_argmax):
    zero_c = value[0].real
    if zero_c == 0.:
        #empty pitch class vector, thus returns white color value.
        #this avoid a nasty divide by 0 error two lines later.
        return (0.,0.)#([0xff]*3
    magn = curr_max/zero_c
    angle = curr_argmax
    return (angle, magn)
    
def center_of_mass_v(utm_max, utm_argmax, utm):
    shape_x, shape_y = np.shape(utm)[:2]
    for y in range(shape_y):
        for x in range(shape_x):
            curr_value = utm[y][x]
            curr_max = utm_max[y][x]
            curr_argmax = utm_argmax[y][x]
            if np.any(curr_value):
                utm_max[y][x] = zeroth_coeff_cm(curr_value, curr_max, curr_argmax)[1]
            for z in range(6):
                utm[y][x][z] = zeroth_coeff_cm(curr_value, curr_value[z], 0)[1]
     
    
    #ignore x axis
    utm_max = np.sum(utm_max, axis=1) #shape (N,)
    weight_sum = 0
    nomi = 0
    for i in range(utm_max.shape[0]):
        nomi += i*utm_max[i]
        weight_sum += utm_max[i]
    y = nomi/weight_sum
    
    return y/(utm_max.shape[0]-1)

def max_utm_to_ws_utm(utm_max, utm_argmax, utm):
    """
    Converts an upper triangle matrix filled with Fourier coefficients into 
    an upper triangle matrix filled with color values that serves as the mathematical model
    holding the color information needed to build the wavescapes plot.
    
    """    
    shape_x, shape_y = np.shape(utm)[:2]
    #RGB => 3 values, RGBA => RGB + 1 value, raw values => angle & magnitude => 2 values
    channel_nbr = 4
    default_value = 0.0
    default_type = np.uint64 
    #+1 to differentiate empty elements from white elements later down the line.
    res = np.full((shape_x, shape_y, channel_nbr), default_value, default_type)
    
    for y in range(shape_y):
        for x in range(shape_x):
            curr_value = utm[y][x]
            curr_max = utm_max[y][x]
            curr_argmax = utm_argmax[y][x]
            if np.any(curr_value):
                angle, magn = zeroth_coeff_cm(curr_value, curr_max, curr_argmax)
                #print(angle, magn)
                res[y][x] = circular_hue_revised(angle, magnitude=magn) 
    
    
    return res