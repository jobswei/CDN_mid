import os
import scipy.io
import cv2
import matplotlib.pyplot as plt

im_root = '/home/wzy/CDN/data/hico_20160224_det/images'
bbox_file = '/home/wzy/CDN/data/hico_20160224_det/anno_bbox.mat'

ld = scipy.io.loadmat(bbox_file)
bbox_train = ld['bbox_train'][0]
bbox_test = ld['bbox_test'][0]
list_action = ld['list_action'][0]

# Change this
i = 100  # image index
j = 0    # hoi index

# Read image
im_file = os.path.join(im_root, 'train2015', bbox_train[i]['filename'][0])
im = cv2.imread(im_file)

# Display image
plt.figure(1)
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
plt.title('Image')

# Display hoi
hoi_id = bbox_train[i]['hoi'][j]['id'][0]
aname = f"{list_action[hoi_id]['vname_ing'][0]} {list_action[hoi_id]['nname'][0]}"
aname = aname.replace('_', ' ')
plt.suptitle(aname)

# Display bbox
if bbox_train[i]['hoi'][j]['invis'][0]:
    print('hoi not visible')
else:
    bboxhuman = bbox_train[i]['hoi'][j]['bboxhuman'][0]
    bboxobject = bbox_train[i]['hoi'][j]['bboxobject'][0]
    connection = bbox_train[i]['hoi'][j]['connection'][0]

    # Define a function visualize_box_conn_one to visualize the bounding boxes and connection
    def visualize_box_conn_one(bboxhuman, bboxobject, connection, color_h, color_o):
        # Implement the visualization logic here
        pass

    visualize_box_conn_one(bboxhuman, bboxobject, connection, 'b', 'g')

plt.show()
