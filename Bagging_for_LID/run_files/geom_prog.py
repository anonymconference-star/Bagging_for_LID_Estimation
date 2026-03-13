import numpy as np
#-----------------------------------------------------------------------------------------------------------------------
#Function to make a simple geometric progression
#(note that for integer sequences, we have to round, and sometimes further modify (round up, or down) for low numbers,
# in order to avoid duplicates, but sometimes even that doesn't work, so we just have to remove duplicates)
def geom_prog(min=1, max=100, step=None, integer=False, remove_duplicates=True):
    def dedupe_keep_order(xs):
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    if step is None:
        a = 2
        step = int(np.log2(max/min)+1)
    else:
        a = (max/min)**(1/(step-1))
    if integer:
        l = [round(min*a**i) for i in range(step)]
    else:
        l = [min*a**i for i in range(step)]
    if remove_duplicates:
        l = dedupe_keep_order(l)
    return l

#A linear progression as well, in case we would want that
def linear_prog(min=1, max=100, step=None, integer=False):
    if step is None:
        a = min
        step = int(max/min)
    else:
        a = (max-min)/(step-1)
    if integer:
        l = [round(min+a*i) for i in range(step)]
    else:
        l = [min+a*i for i in range(step)]
    return l
#-----------------------------------------------------------------------------------------------------------------------
#Progressions for MAIN Paper results

#Effectiveness of bagging/Bagging and smoothing

sr_prog_smoothing_weighing_tests = [0.6, 0.42857142857142855, 0.3, 0.21428571428571427, 0.15789473684210525, 0.11538461538461539, 0.08108108108108109, 0.058823529411764705, 0.041666666666666664]

k_prog_smoothing_weighing_tests = [5, 7, 10, 14, 19, 26, 37, 51, 72]

#Number of bags test (mse bar charts)
Nbag_prog_number_of_bags_tests = [3, 4, 5, 6, 8, 11, 14, 18, 24, 30, 39, 51, 66, 85, 110, 143, 185, 239, 309, 400]

#Sampling rate test (mse bar charts)
sr_prog_sampling_rate_test = [1, 0.8541315, 0.72954061, 0.62312362, 0.53222951, 0.45459399,0.38828304, 0.33164478, 0.28326825, 0.24194833, 0.20665569, 0.17651114, 0.15076372, 0.12877204, 0.10998826, 0.09394443, 0.0802409, 0.06853628, 0.058539, 0.05]

#Interaction of sampling rate and number of bags (mse difference heatmaps)
sr_prog_sampling_rate_number_of_bags_interaction = [1, 0.8541315, 0.72954061, 0.62312362, 0.53222951, 0.45459399,0.38828304, 0.33164478, 0.28326825, 0.24194833, 0.20665569, 0.17651114, 0.15076372, 0.12877204, 0.10998826, 0.09394443, 0.0802409, 0.06853628, 0.058539, 0.05]
Nbag_prog_sampling_rate_number_of_bags_interaction = [1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 16, 19, 24, 29, 36, 44, 54, 66, 81, 100]

#Interaction of sampling rate and k (mse difference heatmaps)
sr_prog_sampling_rate_k_interaction = [1, 0.8541315, 0.72954061, 0.62312362, 0.53222951, 0.45459399, 0.38828304, 0.33164478, 0.28326825, 0.24194833, 0.20665569, 0.17651114, 0.15076372, 0.12877204, 0.10998826, 0.09394443, 0.0802409, 0.06853628, 0.058539, 0.05]
k_prog_sampling_rate_k_interaction = [5, 6, 7, 8, 9, 11, 13, 15, 18, 21, 24, 28, 33, 39, 45, 53, 62, 73, 85, 100]
#-----------------------------------------------------------------------------------------------------------------------
#Progressions for extra supplementary results

#Supplementary parameter tests

#Variable k (for comparing bagging and/or smoothing variants).

k_prog_ = [5, 6, 7, 8, 9, 11, 13, 15, 18, 21, 24, 28, 33, 39, 45, 53, 62, 73, 85, 100]

#Variable sampling rate (for comparing bagging and/or smoothing variants).

sr_prog_ = [0.6, 0.42857142857142855, 0.3, 0.21428571428571427, 0.15789473684210525, 0.11538461538461539, 0.08108108108108109, 0.058823529411764705, 0.041666666666666664]

#Variable number of bags (for comparing bagging and/or smoothing variants).

Nbag_prog_ = [3, 4, 5, 6, 8, 11, 14, 18, 24, 30, 39, 51, 66, 85, 110, 143, 185, 239, 309, 400]

#Variable number of data points (for comparing bagging and/or smoothing variants).

N_prog_ = [100, 123, 151, 185, 228, 280, 344, 423, 519, 638, 784, 963, 1183, 1454, 1786, 2194, 2696, 3312, 4070, 5000]

#Variable LID of uniform dataset (for comparing bagging and/or smoothing variants). There are two tests here, one where embedding dim = LID, and one where embedding dim = max(LID_prog_).

LID_prog_ =  [1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 16, 19, 24, 29, 36, 44, 54, 66, 81, 100]

#Variable t (for analyzing weighted bagging).

t_prog_ = [0.1, 0.11707799137227792, 0.13707256063767184, 0.16048180071713386, 0.18788886879768224, 0.219976513600421, 0.25754408361413866, 0.3015274399935735, 0.35302227018072646, 0.4133113830244106, 0.483896665357962, 0.5665364961185353, 0.6632895500464645, 0.7765660821766207, 0.9091879706907806, 1.064459013883141, 1.2462472324355385, 1.4590812272681322, 1.7082629933755105, 2]

t_prog_small = [0.1, 0.14542154334489538, 0.21147425268811282, 0.3075291220361376, 0.44721359549995787, 0.6503449126242363, 0.9457416090031757, 1.3753120438672641, 1.9999999999999998]

if __name__ == "__main__":
    print(geom_prog(min=0.1, max=2, step=9))
