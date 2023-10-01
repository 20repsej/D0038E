'''
Takes .csv file of the sign gestures and displays what they look like by creating a sketeton
'''

### Imports
import csv
import numpy as np
import matplotlib.pyplot as plt


file = "train-final.csv"

fields = []
rows = []


with open(file, 'r') as csvfile:
    csvreader = csv.reader(csvfile)

    fields = next(csvreader)

    for row in csvreader:
        rows.append(row)

    print("Total no of rows: %d"%(csvreader.line_num))


first_row = rows[0]


bone_structure = [[1,2], [2,3],[2,4],[3,5],[5,7],[7,9],[4,6],[6,8],[8,10],[2,11],[11,12],[12,13],[12,14],[13,15],[15,17],[17,19],[14,16],[16,18],[18,20]]

bone_list = np.array(bone_structure) - 1




meanPos = np.array(first_row[:60])
stdPos = np.array(first_row[61:120])
meanAngle = np.array(first_row[121:180])
stdAngle = np.array(first_row[181:240])


x_mean_pos = np.array(meanPos[::3])
y_mean_pos = np.array(meanPos[1::3])
z_mean_pos = np.array(meanPos[2::3])
x_mean_angle = np.array(meanAngle[::3])
y_mean_angle = np.array(meanAngle[1::3])
z_mean_angle = np.array(meanPos[2::3])

x_mean_pos = x_mean_pos.astype(np.float64)
y_mean_pos = y_mean_pos.astype(np.float64)
z_mean_pos = z_mean_pos.astype(np.float64)
x_mean_angle = x_mean_angle.astype(np.float64)
y_mean_angle = y_mean_angle.astype(np.float64)
z_mean_angle = z_mean_angle.astype(np.float64)


print(x_mean_pos)
print(x_mean_angle)

xs = x_mean_pos * x_mean_angle
ys = y_mean_pos * y_mean_angle
zs = z_mean_pos * z_mean_angle



x_std_pos = np.array(stdPos[::3])
y_std_pos = np.array(stdPos[1::3])
z_std_pos = np.array(stdPos[2::3])
x_std_angle = np.array(stdAngle[::3])
y_std_angle = np.array(stdAngle[1::3])
z_std_angle = np.array(stdPos[2::3])

x_std_pos = x_std_pos.astype(np.float64)
y_std_pos = y_std_pos.astype(np.float64)
z_std_pos = z_std_pos.astype(np.float64)
x_std_angle = x_std_angle.astype(np.float64)
y_std_angle = y_std_angle.astype(np.float64)
z_std_angle = z_std_angle.astype(np.float64)


print(x_std_pos)
print(x_std_angle)

xss = x_std_pos * x_std_angle
yss = y_std_pos * y_std_angle
zss = z_std_pos * z_std_angle


#xt = np.array([5,4,3,7,4,2,8,4,6,9,1,5])
#yt = np.array([9,7,4,3,6,8,5,3,5,6,4,8])

fig, ax = plt.subplots()
#ax = plt.axes(projection='3d')

#ax.scatter(xs, ys, zs)
ax.scatter(x_mean_pos, y_mean_pos)
#ax.scatter(xss, yss)
for bone in bone_list:
    ax.plot([x_mean_pos[bone[0]], x_mean_pos[bone[1]]], [y_mean_pos[bone[0]], y_mean_pos[bone[1]]], 'r')



plt.show()