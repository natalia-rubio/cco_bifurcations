## THIS WAS WRITTEN BY CHATGPT
import numpy as np

def vector_cross_product(v1, v2):
    """Return the cross product of two vectors in 3D."""
    return np.array([
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    ])

def vector_dot_product(v1, v2):
    """Return the dot product of two vectors in 3D."""
    return np.dot(v1, v2)

def vector_subtract(v1, v2):
    """Subtract vector v2 from v1 in 3D."""
    return np.array(v1) - np.array(v2)

def is_point_on_segment(p, a, b):
    """Check if point p is on the line segment ab."""
    ab = vector_subtract(b, a)
    ap = vector_subtract(p, a)
    bp = vector_subtract(p, b)
    
    # Check if the point is collinear with the segment and lies within the bounds
    cross_product = vector_cross_product(ab, ap)
    if not np.allclose(cross_product, 0):  # Not collinear
        return False
    
    dot_product1 = vector_dot_product(ap, ab)
    dot_product2 = vector_dot_product(bp, ab)
    
    return dot_product1 >= 0 and dot_product2 <= 0

def do_segments_intersect(p1, p2, p3, p4):
    """
    Check if 3D line segments p1-p2 and p3-p4 intersect.
    Uses parametric equations of the segments.
    """
    # Direction vectors
    d1 = vector_subtract(p2, p1)
    d2 = vector_subtract(p4, p3)
    
    # Cross product of direction vectors
    cross_d1_d2 = vector_cross_product(d1, d2)
    
    # If the cross product is zero, the segments are parallel (or collinear)
    if np.allclose(cross_d1_d2, [0, 0, 0]):
        # If parallel, check if the segments are collinear and overlap
        #print("Parallel segs") 
        return is_point_on_segment(p1, p3, p4) or is_point_on_segment(p2, p3, p4)#is_point_on_segment(p3, p1, p2)
    
    # Compute the vector between p1 and p3
    p1_to_p3 = vector_subtract(p3, p1)

    # if this is not true, the segments are skew and do not intersect
    if abs(vector_dot_product(p1_to_p3, cross_d1_d2)) > 0.0001:
        #print("No intersection")
        return False
    
    # Compute the scalar factors for the parametric equations of the segments
    # t = vector_dot_product(vector_cross_product(p1_to_p3, d2), cross_d1_d2) / np.linalg.norm(cross_d1_d2)**2
    # u = vector_dot_product(vector_cross_product(p1_to_p3, d1), cross_d1_d2) / np.linalg.norm(cross_d1_d2)**2
    t = vector_dot_product(vector_cross_product(p1_to_p3, d2), cross_d1_d2) / np.linalg.norm(cross_d1_d2)**2
    u = vector_dot_product(vector_cross_product(p1_to_p3, d1), cross_d1_d2) / np.linalg.norm(cross_d1_d2)**2
    # Check if t and u are within the [0, 1] range, which indicates the segments intersect
    return 0 <= t <= 1 and 0 <= u <= 1

# def do_segments_intersect(p1, p2, p3, p4):
#     if np.linalg.norm(vector_subtract(p3,p1)) < np.linalg.norm(vector_subtract(p2,p1)):
#         return True
#     if np.linalg.norm(vector_subtract(p4,p1)) < np.linalg.norm(vector_subtract(p2,p1)):
#         return True
#     if np.linalg.norm(vector_subtract(p3,p2)) < np.linalg.norm(vector_subtract(p2,p1)):
#         return True
#     if np.linalg.norm(vector_subtract(p4,p2)) < np.linalg.norm(vector_subtract(p2,p1)):
#         return True
#     return False

def check_3d_curve_intersection(curve1, curve2):
    """
    Check if two 3D curves, each defined by a list of 3D points, intersect.
    curve1 and curve2 are lists of (x, y, z) tuples.
    """
    for i in range(len(curve1) - 1):
        for j in range(len(curve2) - 1):
            # Check if the segment (curve1[i], curve1[i+1]) intersects with (curve2[j], curve2[j+1])
            
            if do_segments_intersect(curve1[i], curve1[i+1], curve2[j], curve2[j+1]):
                print("Intersection segments")
                #print("Intersection at {i} {curve1[i]}-{curve1[i+1]} and {j} {curve2[j]}-{curve2[j+1]}".format(i=i, j=j))
                return True
    return False

# Example usage:
curve1 = [(1, 1, 1), (3, 3, 1), (5, 1, 1)]
curve2 = [(2, 2, 1), (4, 0, 1), (6, 2, 1)]

print(check_3d_curve_intersection(curve1, curve2))  # Output: True or False based on the curves
