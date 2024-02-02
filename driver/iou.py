from shapely.geometry import Polygon

def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area/poly_1.union(poly_2).area
    return iou
    
box_1 = [[0,0],[0,0],[0,0],[0,0]]
box_2 = [[320, 451], [608, 451], [608, 1061], [320, 1061]]


print(calculate_iou(box_1,box_2))
