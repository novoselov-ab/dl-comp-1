import numpy as np

def bbox2(img):
    rows = np.where(np.any(img, axis=1))
    cols = np.where(np.any(img, axis=0))
    if rows[0].size == 0 or cols[0].size == 0:
        return (0, 0, 0, 0)
    rmin, rmax = rows[0][[0, -1]]
    cmin, cmax = cols[0][[0, -1]]

    #(y0,y1,x0,x1)
    return rmin, rmax, cmin, cmax

def bbox2_unit(img):
    (y0,y1,x0,x1) = bbox2(img)
    w = float(img.shape[0])
    h = float(img.shape[1])
    return (y0/h, y1/h, x0/w, x1/w)
