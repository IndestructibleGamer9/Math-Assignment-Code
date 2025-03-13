def intersect(a, b, c, x):
    """
    Calculate the value of a quadratic function at point x.
    
    Args:
        a (float): Coefficient of x²
        b (float): Coefficient of x
        c (float): Constant term
        x (float): X-coordinate
    
    Returns:
        float: Y-coordinate of the quadratic function at point x
    """
    return (a * (x**2) + b * x + c)

def find_all_points(a, b, c):
    """
    Generate y-coordinates for x values from 0 to 26 using quadratic function.
    
    Args:
        a (float): Coefficient of x²
        b (float): Coefficient of x
        c (float): Constant term
    
    Returns:
        list: List of y-coordinates
    """
    points = []
    for x in range(27):  # Simplified loop from 0 to 26
        points.append(intersect(a, b, c, x))
    return points

def find_area(e, intersects, w):
    """
    Calculate area using formula A = W/2(E+2M).
    
    Args:
        e (float): E value in formula
        intersects (list): List of intersection points
        w (float): Width value
    
    Returns:
        float: Calculated area
    """
    m = sum(intersects)
    return (w/2) * (e + 2*m)

def find_distance(list1, list2):
    """
    Calculate the difference between corresponding elements of two lists.
    
    Args:
        list1 (list): First list of numbers
        list2 (list): Second list of numbers
    
    Returns:
        list: List of differences (list2[i] - list1[i])
    """
    return [y - x for x, y in zip(list1, list2)]  # Using list comprehension and zip

# Constants for the quadratic function coefficients
A = -499/17940
B = 13457/17940
C1 = 3.3
C2 = 4.3

# Calculate points for both parabolas
lower_parabola = find_all_points(A, B, C1)
higher_parabola = find_all_points(A, B, C2)

# Calculate intersections and final area
intersections = find_distance(lower_parabola, higher_parabola)
area = find_area(0, intersections, 1)
print(f"Calculated area: {area}")