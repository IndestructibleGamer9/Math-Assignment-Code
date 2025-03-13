import unittest
from main import intersect, find_all_points, find_area, find_distance

class TestMathMethods(unittest.TestCase):

    def test_intersect(self):
        self.assertEqual(intersect(1, 0, 0, 2), 4)
        self.assertEqual(intersect(1, 1, 1, 1), 3)
        self.assertEqual(intersect(0, 0, 0, 0), 0)

    def test_find_all_points(self):
        self.assertEqual(find_all_points(0, 0, 1), [1]*27)
        self.assertEqual(find_all_points(1, 0, 0), [x**2 for x in range(27)])
        self.assertEqual(find_all_points(0, 1, 0), [x for x in range(27)])

    def test_find_area(self):
        self.assertEqual(find_area(0, [1, 1, 1], 1), 3)
        self.assertEqual(find_area(1, [1, 1, 1], 2), 7)
        self.assertEqual(find_area(0, [0, 0, 0], 1), 0)
    
    def test_find_distance(self):
        self.assertEqual(find_distance([1, 2, 3], [4, 5, 6]), [3, 3, 3])
        self.assertEqual(find_distance([0, 0, 0], [0, 0, 0]), [0, 0, 0])
        self.assertEqual(find_distance([-1, -2, -3], [1, 2, 3]), [2, 4, 6])
        self.assertEqual(find_distance([1.5, 2.5, 3.5], [4.5, 5.5, 6.5]), [3.0, 3.0, 3.0])

if __name__ == '__main__':
    unittest.main()