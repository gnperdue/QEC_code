'''
Usage:
    python test_gates.py
'''
import unittest

import logging
LOGGER = logging.getLogger(__name__)

import sys
sys.path.append('..')   # the `general_qec` package sits above us

import numpy as np
from general_qec.gates import sigma_I, sigma_x, sigma_y, sigma_z
from general_qec.gates import adj_CNOT, flipped_adj_CNOT, small_non_adj_CNOT
from general_qec.gates import non_adj_CNOT, flipped_non_adj_CNOT, CNOT
from general_qec.gates import cnot, flipped_cnot
from general_qec.gates import adj_CZ, non_adj_CZ, CZ


class TestGates(unittest.TestCase):

    def setUp(self) -> None:
        self.three_qubit000 = np.array([[1], [0], [0], [0], [0], [0], [0], [0]])
        self.three_qubit001 = np.array([[0], [1], [0], [0], [0], [0], [0], [0]])
        self.three_qubit010 = np.array([[0], [0], [1], [0], [0], [0], [0], [0]])
        self.three_qubit011 = np.array([[0], [0], [0], [1], [0], [0], [0], [0]])
        self.three_qubit100 = np.array([[0], [0], [0], [0], [1], [0], [0], [0]])
        self.three_qubit101 = np.array([[0], [0], [0], [0], [0], [1], [0], [0]])
        self.three_qubit110 = np.array([[0], [0], [0], [0], [0], [0], [1], [0]])
        self.three_qubit111 = np.array([[0], [0], [0], [0], [0], [0], [0], [1]])

        return super().setUp()

    def test_commutators(self):
        LOGGER.info(sys._getframe().f_code.co_name)
        self.assertTrue(np.all(
            np.matmul(sigma_x, sigma_y) - np.matmul(sigma_y, sigma_x) == -2j*sigma_z
        ))
        self.assertTrue(np.all(
            np.matmul(sigma_y, sigma_z) - np.matmul(sigma_z, sigma_y) == -2j*sigma_x
        ))
        self.assertTrue(np.all(
            np.matmul(sigma_z, sigma_x) - np.matmul(sigma_x, sigma_z) == -2j*sigma_y
        ))

    def test_CNOT(self):
        LOGGER.info(sys._getframe().f_code.co_name)
        self.assertEqual(adj_CNOT(0, 1, 4).shape, (16, 16))
        self.assertEqual(adj_CNOT(2, 3, 4).shape, (16, 16))
        self.assertTrue(np.all(
            np.matmul(adj_CNOT(0, 1, 3), self.three_qubit100) == \
                self.three_qubit110
        ))
        self.assertTrue(np.all(
            np.matmul(adj_CNOT(1, 2, 3), self.three_qubit011) == \
                self.three_qubit010
        ))
        self.assertEqual(flipped_adj_CNOT(1, 0, 4).shape, (16, 16))
        self.assertEqual(flipped_adj_CNOT(3, 2, 4).shape, (16, 16))
        self.assertTrue(np.all(
            np.matmul(flipped_adj_CNOT(1, 0, 3), self.three_qubit010) == \
                self.three_qubit110
        ))
        self.assertTrue(np.all(
            np.matmul(flipped_adj_CNOT(2, 1, 3), self.three_qubit111) == \
                self.three_qubit101
        ))
        self.assertEqual(small_non_adj_CNOT().shape, (8, 8))
        self.assertEqual(non_adj_CNOT(0, 2, 4).shape, (16, 16))
        self.assertEqual(non_adj_CNOT(1, 3, 4).shape, (16, 16))
        self.assertTrue(np.all(
            np.matmul(non_adj_CNOT(0, 2, 3), self.three_qubit110) == \
                self.three_qubit111
        ))
        self.assertTrue(np.all(
            np.matmul(flipped_non_adj_CNOT(2, 0, 3), self.three_qubit001) == \
                self.three_qubit101
        ))
        self.assertTrue(np.all(
            CNOT(0, 1, 2) == cnot
        ))
        self.assertTrue(np.all(
            CNOT(1, 0, 2) == flipped_cnot
        ))
        self.assertTrue(np.all(
            adj_CNOT(2, 3, 4) == CNOT(2, 3, 4)
        ))
        self.assertTrue(np.all(
            flipped_adj_CNOT(2, 1, 3) == CNOT(2, 1, 3)
        ))
        self.assertTrue(np.all(
            non_adj_CNOT(0, 3, 5) == CNOT(0, 3, 5)
        ))
        self.assertTrue(np.all(
            flipped_non_adj_CNOT(4, 0, 5) == CNOT(4, 0, 5)
        ))

    def test_CZ(self):
        LOGGER.info(sys._getframe().f_code.co_name)
        self.assertTrue(np.all(
            adj_CZ(0, 1, 2) == CZ(0, 1, 2)
        ))
        self.assertTrue(np.all(
            adj_CZ(3, 4, 5) == CZ(3, 4, 5)
        ))
        self.assertTrue(np.all(
            non_adj_CZ(0, 4, 5) == CZ(0, 4, 5)
        ))
        self.assertTrue(np.all(
            non_adj_CZ(3, 0, 5) == CZ(3, 0, 5)
        ))


if __name__ == '__main__':
    unittest.main()


