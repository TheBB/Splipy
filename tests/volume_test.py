from __future__ import annotations

import unittest
from math import pi, sqrt

import numpy as np

import splipy.volume_factory as vf
from splipy import BSplineBasis, Volume


class TestVolume(unittest.TestCase):
    def test_evaluate(self):
        # creating the identity mapping by different size for all directions
        vol = Volume(BSplineBasis(7), BSplineBasis(6), BSplineBasis(5))

        # call evaluation at a 2x3x4 grid of points
        u_val = np.linspace(0, 1, 2)
        v_val = np.linspace(0, 1, 3)
        w_val = np.linspace(0, 1, 4)
        value = vol(u_val, v_val, w_val)
        self.assertEqual(value.shape[0], 2)  # 2 u-evaluation points
        self.assertEqual(value.shape[1], 3)  # 3 v-evaluation points
        self.assertEqual(value.shape[2], 4)  # 4 w-evaluation points
        self.assertEqual(value.shape[3], 3)  # 3 dimensions (x,y,z)
        self.assertEqual(vol.order(), (7, 6, 5))
        for i, u in enumerate(u_val):
            for j, v in enumerate(v_val):
                for k, w in enumerate(w_val):
                    self.assertAlmostEqual(value[i, j, k, 0], u)  # identity map x=u
                    self.assertAlmostEqual(value[i, j, k, 1], v)  # identity map y=v
                    self.assertAlmostEqual(value[i, j, k, 2], w)  # identity map z=w

        # test errors and exceptions
        with self.assertRaises(ValueError):
            vol(-10, 0.5, 0.5)  # evalaute outside parametric domain
        with self.assertRaises(ValueError):
            vol(+10, 0.3, 0.3)  # evalaute outside parametric domain
        with self.assertRaises(ValueError):
            vol(0.5, -10, 0.123)  # evalaute outside parametric domain
        with self.assertRaises(ValueError):
            vol(0.5, +10, 0.123)  # evalaute outside parametric domain
        with self.assertRaises(ValueError):
            vol(0.5, 0.2, +10)  # evalaute outside parametric domain
        with self.assertRaises(ValueError):
            vol(0.5, 0.2, -10)  # evalaute outside parametric domain

    def test_evaluate_nontensor(self):
        vol = Volume(BSplineBasis(7), BSplineBasis(7), BSplineBasis(5))

        u_val = [0, 0.1, 0.9, 0.3]
        v_val = [0.2, 0.3, 0.9, 0.4]
        w_val = [0.3, 0.5, 0.5, 0.0]
        value = vol(u_val, v_val, w_val, tensor=False)

        self.assertEqual(value.shape[0], 4)
        self.assertEqual(value.shape[1], 3)

        for i, (u, v, w) in enumerate(zip(u_val, v_val, w_val)):
            self.assertAlmostEqual(value[i, 0], u)  # identity map x=u
            self.assertAlmostEqual(value[i, 1], v)  # identity map y=v
            self.assertAlmostEqual(value[i, 2], w)  # identity map z=w

    def test_indexing(self):
        v = Volume()

        self.assertEqual(v[0][0], 0.0)
        self.assertEqual(v[0][1], 0.0)
        self.assertEqual(v[0][2], 0.0)
        self.assertEqual(v[1][0], 1.0)
        self.assertEqual(v[1][1], 0.0)
        self.assertEqual(v[1][2], 0.0)
        self.assertEqual(v[-1][0], 1.0)
        self.assertEqual(v[-1][1], 1.0)
        self.assertEqual(v[-1][2], 1.0)

        self.assertEqual(v[:][0, 0], 0.0)
        self.assertEqual(v[:][1, 0], 1.0)
        self.assertEqual(v[:][1, 1], 0.0)
        self.assertEqual(v[:][2, 1], 1.0)
        self.assertEqual(v[:][2, 2], 0.0)
        self.assertEqual(v[:][5, 0], 1.0)
        self.assertEqual(v[:][5, 1], 0.0)
        self.assertEqual(v[:][5, 2], 1.0)

        self.assertEqual(v[0, 0, 0][0], 0.0)
        self.assertEqual(v[0, 0, 0][1], 0.0)
        self.assertEqual(v[0, 0, 0][2], 0.0)
        self.assertEqual(v[0, 1, 0][0], 0.0)
        self.assertEqual(v[0, 1, 0][1], 1.0)
        self.assertEqual(v[0, 1, 0][2], 0.0)
        self.assertEqual(v[0, :, 1][0, 0], 0.0)
        self.assertEqual(v[0, :, 1][0, 1], 0.0)
        self.assertEqual(v[0, :, 1][0, 2], 1.0)
        self.assertEqual(v[0, :, 1][1, 0], 0.0)
        self.assertEqual(v[0, :, 1][1, 1], 1.0)
        self.assertEqual(v[0, :, 1][1, 2], 1.0)

    def test_raise_order(self):
        # more or less random 3D volume with p=[2,2,1] and n=[4,3,2]
        controlpoints = [
            [0, 0, 0],
            [-1, 1, 0],
            [0, 2, 0],
            [1, -1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [2, 1, 0],
            [2, 2, 0],
            [2, 3, 0],
            [3, 0, 0],
            [4, 1, 0],
            [3, 2, 0],
            [0, 0, 1],
            [-1, 1, 1],
            [0, 2, 1],
            [1, -1, 2],
            [1, 0, 2],
            [1, 1, 2],
            [2, 1, 2],
            [2, 2, 2],
            [2, 3, 2],
            [3, 0, 1],
            [4, 1, 1],
            [3, 2, 1],
        ]
        basis1 = BSplineBasis(3, [0, 0, 0, 0.4, 1, 1, 1])
        basis2 = BSplineBasis(3, [0, 0, 0, 1, 1, 1])
        basis3 = BSplineBasis(2, [0, 0, 1, 1])
        vol = Volume(basis1, basis2, basis3, controlpoints)

        self.assertEqual(vol.order(), (3, 3, 2))
        evaluation_point1 = vol(0.23, 0.37, 0.44)  # pick some evaluation point (could be anything)

        vol.raise_order(1, 2, 4)

        self.assertEqual(vol.order(), (4, 5, 6))
        evaluation_point2 = vol(0.23, 0.37, 0.44)

        # evaluation before and after RaiseOrder should remain unchanged
        self.assertAlmostEqual(evaluation_point1[0], evaluation_point2[0])
        self.assertAlmostEqual(evaluation_point1[1], evaluation_point2[1])
        self.assertAlmostEqual(evaluation_point1[2], evaluation_point2[2])

        # test a rational 3D volume
        controlpoints = [
            [0, 0, 1, 1],
            [-1, 1, 0.96, 1],
            [0, 2, 1, 1],
            [1, -1, 1, 1],
            [1, 0, 0.8, 1],
            [1, 1, 1, 1],
            [2, 1, 0.89, 1],
            [2, 2, 0.9, 1],
            [2, 3, 1, 1],
            [3, 0, 1, 1],
            [4, 1, 1, 1],
            [3, 2, 1, 1],
            [0, 0, 1, 2],
            [-1, 1, 0.7, 2],
            [0, 2, 1.3, 2],
            [1, -1, 1, 2],
            [1, 0, 0.77, 2],
            [1, 1, 1, 2],
            [2, 1, 0.89, 1],
            [2, 2, 0.8, 4],
            [2, 3, 1, 1],
            [3, 0, 1, 1],
            [4, 1, 1, 1],
            [3, 2, 1, 1],
        ]
        basis1 = BSplineBasis(3, [0, 0, 0, 0.4, 1, 1, 1])
        basis2 = BSplineBasis(3, [0, 0, 0, 1, 1, 1])
        basis3 = BSplineBasis(2, [0, 0, 1, 1])
        vol = Volume(basis1, basis2, basis3, controlpoints, True)

        self.assertEqual(vol.order()[0], 3)
        self.assertEqual(vol.order()[1], 3)
        evaluation_point1 = vol(0.23, 0.37, 0.44)

        vol.raise_order(1, 2, 1)

        self.assertEqual(vol.order(), (4, 5, 3))
        evaluation_point2 = vol(0.23, 0.37, 0.44)

        # evaluation before and after RaiseOrder should remain unchanged
        self.assertAlmostEqual(evaluation_point1[0], evaluation_point2[0])
        self.assertAlmostEqual(evaluation_point1[1], evaluation_point2[1])
        self.assertAlmostEqual(evaluation_point1[2], evaluation_point2[2])

    def test_lower_order(self):
        b = BSplineBasis(4, [0, 0, 0, 0, 0.2, 0.3, 0.3, 0.6, 0.9, 1, 1, 1, 1])
        t = b.greville()
        Y, X, Z = np.meshgrid(t, t, t)
        cp = np.zeros((len(t), len(t), len(t), 3))
        cp[..., 0] = X * (1 - Y)
        cp[..., 1] = X * X
        cp[..., 2] = Z**2 + 2

        vol = vf.interpolate(cp, [b, b, b])
        vol2 = vol.lower_order(1)  # still in space, vol2 is *also* exact
        u = np.linspace(0, 1, 5)
        v = np.linspace(0, 1, 6)
        w = np.linspace(0, 1, 7)
        self.assertTrue(np.allclose(vol(u, v, w), vol2(u, v, w)))
        self.assertTupleEqual(vol.order(), (4, 4, 4))
        self.assertTupleEqual(vol2.order(), (3, 3, 3))

    def test_insert_knot(self):
        # more or less random 3D volume with p=[2,2,1] and n=[4,3,2]
        controlpoints = [
            [0, 0, 0],
            [-1, 1, 0],
            [0, 2, 0],
            [1, -1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [2, 1, 0],
            [2, 2, 0],
            [2, 3, 0],
            [3, 0, 0],
            [4, 1, 0],
            [3, 2, 0],
            [0, 0, 1],
            [-1, 1, 1],
            [0, 2, 1],
            [1, -1, 2],
            [1, 0, 2],
            [1, 1, 2],
            [2, 1, 2],
            [2, 2, 2],
            [2, 3, 2],
            [3, 0, 1],
            [4, 1, 1],
            [3, 2, 1],
        ]
        basis1 = BSplineBasis(3, [0, 0, 0, 0.4, 1, 1, 1])
        basis2 = BSplineBasis(3, [0, 0, 0, 1, 1, 1])
        basis3 = BSplineBasis(2, [0, 0, 1, 1])
        vol = Volume(basis1, basis2, basis3, controlpoints)

        evaluation_point1 = vol(0.23, 0.37, 0.44)  # pick some evaluation point (could be anything)

        vol.insert_knot(0.20, 0)
        vol.insert_knot(0.5, "u")
        vol.insert_knot(0.7, 0)
        vol.insert_knot(0.1, 1)
        vol.insert_knot(1.0 / 3, 1)
        vol.insert_knot(0.8, 2)
        vol.insert_knot(0.9, "W")
        knot1, knot2, knot3 = vol.knots(with_multiplicities=True)
        self.assertEqual(len(knot1), 10)  # 7 to start with, 3 new ones
        self.assertEqual(len(knot2), 8)  # 6 to start with, 2 new ones
        self.assertEqual(len(knot3), 6)  # 4 to start with, 2 new ones
        self.assertEqual(vol.controlpoints.shape, (7, 5, 4, 3))

        evaluation_point2 = vol(0.23, 0.37, 0.44)

        # evaluation before and after insert_knot should remain unchanged
        self.assertAlmostEqual(evaluation_point1[0], evaluation_point2[0])
        self.assertAlmostEqual(evaluation_point1[1], evaluation_point2[1])
        self.assertAlmostEqual(evaluation_point1[2], evaluation_point2[2])

        # test a rational 3D volume
        controlpoints = [
            [0, 0, 1, 1],
            [-1, 1, 0.96, 1],
            [0, 2, 1, 1],
            [1, -1, 1, 1],
            [1, 0, 0.8, 1],
            [1, 1, 1, 1],
            [2, 1, 0.89, 1],
            [2, 2, 0.9, 1],
            [2, 3, 1, 1],
            [3, 0, 1, 1],
            [4, 1, 1, 1],
            [3, 2, 1, 1],
            [0, 0, 1, 2],
            [-1, 1, 0.7, 2],
            [0, 2, 1.3, 2],
            [1, -1, 1, 2],
            [1, 0, 0.77, 2],
            [1, 1, 1, 2],
            [2, 1, 0.89, 1],
            [2, 2, 0.8, 4],
            [2, 3, 1, 1],
            [3, 0, 1, 1],
            [4, 1, 1, 1],
            [3, 2, 1, 1],
        ]
        basis1 = BSplineBasis(3, [0, 0, 0, 0.4, 1, 1, 1])
        basis2 = BSplineBasis(3, [0, 0, 0, 1, 1, 1])
        basis3 = BSplineBasis(2, [0, 0, 1, 1])
        vol = Volume(basis1, basis2, basis3, controlpoints, True)

        evaluation_point1 = vol(0.23, 0.37, 0.44)

        vol.insert_knot([0.20, 0.5, 0.7], 0)
        vol.insert_knot([0.1, 1.0 / 3], 1)
        vol.insert_knot([0.8, 0.9], 2)
        knot1, knot2, knot3 = vol.knots(with_multiplicities=True)
        self.assertEqual(len(knot1), 10)  # 7 to start with, 3 new ones
        self.assertEqual(len(knot2), 8)  # 6 to start with, 2 new ones
        self.assertEqual(len(knot3), 6)  # 4 to start with, 2 new ones
        self.assertEqual(vol.controlpoints.shape, (7, 5, 4, 4))

        evaluation_point2 = vol(0.23, 0.37, 0.44)

        # evaluation before and after RaiseOrder should remain unchanged
        self.assertAlmostEqual(evaluation_point1[0], evaluation_point2[0])
        self.assertAlmostEqual(evaluation_point1[1], evaluation_point2[1])
        self.assertAlmostEqual(evaluation_point1[2], evaluation_point2[2])

    def test_force_rational(self):
        # more or less random 3D volume with p=[3,2,1] and n=[4,3,2]
        controlpoints = [
            [0, 0, 1],
            [-1, 1, 1],
            [0, 2, 1],
            [1, -1, 2],
            [1, 0, 2],
            [1, 1, 2],
            [2, 1, 2],
            [2, 2, 2],
            [2, 3, 2],
            [3, 0, 0],
            [4, 1, 0],
            [3, 2, 0],
            [0, 0, 3],
            [-1, 1, 3],
            [0, 2, 3],
            [1, -1, 5],
            [1, 0, 5],
            [1, 1, 5],
            [2, 1, 4],
            [2, 2, 4],
            [2, 3, 4],
            [3, 0, 2],
            [4, 1, 2],
            [3, 2, 2],
        ]
        basis1 = BSplineBasis(4, [0, 0, 0, 0, 2, 2, 2, 2])
        basis2 = BSplineBasis(3, [0, 0, 0, 1, 1, 1])
        basis3 = BSplineBasis(2, [0, 0, 1, 1])
        vol = Volume(basis1, basis2, basis3, controlpoints)

        evaluation_point1 = vol(0.23, 0.66, 0.32)
        control_point1 = vol[0]
        vol.force_rational()
        evaluation_point2 = vol(0.23, 0.66, 0.32)
        control_point2 = vol[0]
        # ensure that volume has not chcanged, by comparing evaluation of it
        self.assertAlmostEqual(evaluation_point1[0], evaluation_point2[0])
        self.assertAlmostEqual(evaluation_point1[1], evaluation_point2[1])
        self.assertAlmostEqual(evaluation_point1[2], evaluation_point2[2])
        # ensure that we include rational weights of 1
        self.assertEqual(len(control_point1), 3)
        self.assertEqual(len(control_point2), 4)
        self.assertEqual(control_point2[3], 1)
        self.assertEqual(vol.rational, True)

    def test_swap(self):
        # more or less random 3D volume with p=[3,2,1] and n=[4,3,2]
        controlpoints = [
            [0, 0, 1],
            [-1, 1, 1],
            [0, 2, 1],
            [1, -1, 2],
            [1, 0, 2],
            [1, 1, 2],
            [2, 1, 2],
            [2, 2, 2],
            [2, 3, 2],
            [3, 0, 0],
            [4, 1, 0],
            [3, 2, 0],
            [0, 0, 3],
            [-1, 1, 3],
            [0, 2, 3],
            [1, -1, 5],
            [1, 0, 5],
            [1, 1, 5],
            [2, 1, 4],
            [2, 2, 4],
            [2, 3, 4],
            [3, 0, 2],
            [4, 1, 2],
            [3, 2, 2],
        ]
        basis1 = BSplineBasis(4, [0, 0, 0, 0, 2, 2, 2, 2])
        basis2 = BSplineBasis(3, [0, 0, 0, 1, 1, 1])
        basis3 = BSplineBasis(2, [0, 0, 1, 1])
        vol = Volume(basis1, basis2, basis3, controlpoints)

        evaluation_point1 = vol(0.23, 0.56, 0.12)
        control_point1 = vol[1]  # this is control point i=(1,0,0), when n=(4,3,2)
        self.assertEqual(vol.order(), (4, 3, 2))
        vol.swap(0, 1)
        evaluation_point2 = vol(0.56, 0.23, 0.12)
        control_point2 = vol[3]  # this is control point i=(0,1,0), when n=(3,4,2)
        self.assertEqual(vol.order(), (3, 4, 2))

        # ensure that volume has not chcanged, by comparing evaluation of it
        self.assertAlmostEqual(evaluation_point1[0], evaluation_point2[0])
        self.assertAlmostEqual(evaluation_point1[1], evaluation_point2[1])
        self.assertAlmostEqual(evaluation_point1[2], evaluation_point2[2])

        # check that the control points have re-ordered themselves
        self.assertEqual(control_point1[0], control_point2[0])
        self.assertEqual(control_point1[1], control_point2[1])
        self.assertEqual(control_point1[2], control_point2[2])

        vol.swap(1, 2)
        evaluation_point3 = vol(0.56, 0.12, 0.23)
        control_point3 = vol[6]  # this is control point i=(0,0,1), when n=(3,2,4)
        self.assertEqual(vol.order(), (3, 2, 4))

        # ensure that volume has not chcanged, by comparing evaluation of it
        self.assertAlmostEqual(evaluation_point1[0], evaluation_point3[0])
        self.assertAlmostEqual(evaluation_point1[1], evaluation_point3[1])
        self.assertAlmostEqual(evaluation_point1[2], evaluation_point3[2])

        # check that the control points have re-ordered themselves
        self.assertEqual(control_point1[0], control_point3[0])
        self.assertEqual(control_point1[1], control_point3[1])
        self.assertEqual(control_point1[2], control_point3[2])

    def test_split(self):
        # more or less random 3D volume with p=[3,2,1] and n=[4,3,2]
        controlpoints = [
            [0, 0, 1],
            [-1, 1, 1],
            [0, 2, 1],
            [1, -1, 2],
            [1, 0, 2],
            [1, 1, 2],
            [2, 1, 2],
            [2, 2, 2],
            [2, 3, 2],
            [3, 0, 0],
            [4, 1, 0],
            [3, 2, 0],
            [0, 0, 3],
            [-1, 1, 3],
            [0, 2, 3],
            [1, -1, 5],
            [1, 0, 5],
            [1, 1, 5],
            [2, 1, 4],
            [2, 2, 4],
            [2, 3, 4],
            [3, 0, 2],
            [4, 1, 2],
            [3, 2, 2],
        ]
        basis1 = BSplineBasis(4, [0, 0, 0, 0, 2, 2, 2, 2])
        basis2 = BSplineBasis(3, [0, 0, 0, 1, 1, 1])
        basis3 = BSplineBasis(2, [0, 0, 1, 1])
        vol = Volume(basis1, basis2, basis3, controlpoints)
        split_u_vol = vol.split([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 0)
        split_v_vol = vol.split(0.1, 1)
        split_w_vol = vol.split([0.4, 0.5, 0.6], 2)

        self.assertEqual(len(split_u_vol), 7)
        self.assertEqual(len(split_v_vol), 2)
        self.assertEqual(len(split_w_vol), 4)

        # check that the u-vector is properly split
        self.assertAlmostEqual(split_u_vol[0].start(0), 0.0)
        self.assertAlmostEqual(split_u_vol[0].end(0), 0.1)
        self.assertAlmostEqual(split_u_vol[1].start(0), 0.1)
        self.assertAlmostEqual(split_u_vol[1].end(0), 0.2)
        self.assertAlmostEqual(split_u_vol[2].start(0), 0.2)
        self.assertAlmostEqual(split_u_vol[2].end(0), 0.3)
        self.assertAlmostEqual(split_u_vol[6].start(0), 0.6)
        self.assertAlmostEqual(split_u_vol[6].end(0), 2.0)
        # check that the other vectors remain unchanged
        self.assertAlmostEqual(split_u_vol[2].start(1), 0.0)
        self.assertAlmostEqual(split_u_vol[2].end(1), 1.0)
        self.assertAlmostEqual(split_u_vol[2].start(2), 0.0)
        self.assertAlmostEqual(split_u_vol[2].end(2), 1.0)
        # check that the v-vector is properly split
        self.assertAlmostEqual(split_v_vol[0].start(1), 0.0)
        self.assertAlmostEqual(split_v_vol[0].end(1), 0.1)
        self.assertAlmostEqual(split_v_vol[1].start(1), 0.1)
        self.assertAlmostEqual(split_v_vol[1].end(1), 1.0)
        # check that the others remain unchanged
        self.assertAlmostEqual(split_v_vol[1].start(0), 0.0)
        self.assertAlmostEqual(split_v_vol[1].end(0), 2.0)
        self.assertAlmostEqual(split_v_vol[1].start(2), 0.0)
        self.assertAlmostEqual(split_v_vol[1].end(2), 1.0)
        # check that the w-vector is properly split
        self.assertAlmostEqual(split_w_vol[1].start(2), 0.4)
        self.assertAlmostEqual(split_w_vol[1].end(2), 0.5)
        self.assertAlmostEqual(split_w_vol[2].start(2), 0.5)
        self.assertAlmostEqual(split_w_vol[2].end(2), 0.6)
        # check that the others remain unchanged
        self.assertAlmostEqual(split_w_vol[1].start(0), 0.0)
        self.assertAlmostEqual(split_w_vol[1].end(0), 2.0)
        self.assertAlmostEqual(split_w_vol[1].start(1), 0.0)
        self.assertAlmostEqual(split_w_vol[1].end(1), 1.0)

        # check that evaluations remain unchanged
        pt1 = vol(0.23, 0.12, 0.3)

        self.assertAlmostEqual(split_u_vol[2].evaluate(0.23, 0.12, 0.3)[0], pt1[0])
        self.assertAlmostEqual(split_u_vol[2].evaluate(0.23, 0.12, 0.3)[1], pt1[1])
        self.assertAlmostEqual(split_u_vol[2].evaluate(0.23, 0.12, 0.3)[2], pt1[2])

        self.assertAlmostEqual(split_v_vol[1].evaluate(0.23, 0.12, 0.3)[0], pt1[0])
        self.assertAlmostEqual(split_v_vol[1].evaluate(0.23, 0.12, 0.3)[1], pt1[1])
        self.assertAlmostEqual(split_v_vol[1].evaluate(0.23, 0.12, 0.3)[2], pt1[2])

        self.assertAlmostEqual(split_w_vol[0].evaluate(0.23, 0.12, 0.3)[0], pt1[0])
        self.assertAlmostEqual(split_w_vol[0].evaluate(0.23, 0.12, 0.3)[1], pt1[1])
        self.assertAlmostEqual(split_w_vol[0].evaluate(0.23, 0.12, 0.3)[2], pt1[2])

    def test_reparam(self):
        # identity mapping, control points generated from knot vector
        basis1 = BSplineBasis(4, [2, 2, 2, 2, 3, 6, 7, 7, 7, 7])
        basis2 = BSplineBasis(3, [-3, -3, -3, 20, 30, 31, 31, 31])
        basis3 = BSplineBasis(5, [0, 0, 0, 0, 0, 8, 8, 8, 8, 8])
        vol = Volume(basis1, basis2, basis3)

        self.assertAlmostEqual(vol.start(0), 2)
        self.assertAlmostEqual(vol.end(0), 7)
        self.assertAlmostEqual(vol.start(1), -3)
        self.assertAlmostEqual(vol.end(1), 31)
        self.assertAlmostEqual(vol.start(2), 0)
        self.assertAlmostEqual(vol.end(2), 8)

        vol.reparam((4, 10), (0, 9), (2, 3))
        self.assertAlmostEqual(vol.start(0), 4)
        self.assertAlmostEqual(vol.end(0), 10)
        self.assertAlmostEqual(vol.start(1), 0)
        self.assertAlmostEqual(vol.end(1), 9)
        self.assertAlmostEqual(vol.start(2), 2)
        self.assertAlmostEqual(vol.end(2), 3)

        vol.reparam((5, 11), direction=0)
        self.assertAlmostEqual(vol.start(0), 5)
        self.assertAlmostEqual(vol.end(0), 11)
        self.assertAlmostEqual(vol.start(1), 0)
        self.assertAlmostEqual(vol.end(1), 9)
        self.assertAlmostEqual(vol.start(2), 2)
        self.assertAlmostEqual(vol.end(2), 3)

        vol.reparam((5, 11), direction=1)
        self.assertAlmostEqual(vol.start(0), 5)
        self.assertAlmostEqual(vol.end(0), 11)
        self.assertAlmostEqual(vol.start(1), 5)
        self.assertAlmostEqual(vol.end(1), 11)
        self.assertAlmostEqual(vol.start(2), 2)
        self.assertAlmostEqual(vol.end(2), 3)

        vol.reparam((5, 11), direction=2)
        self.assertAlmostEqual(vol.start(0), 5)
        self.assertAlmostEqual(vol.end(0), 11)
        self.assertAlmostEqual(vol.start(1), 5)
        self.assertAlmostEqual(vol.end(1), 11)
        self.assertAlmostEqual(vol.start(2), 5)
        self.assertAlmostEqual(vol.end(2), 11)

        vol.reparam((-9, 9))
        self.assertAlmostEqual(vol.start(0), -9)
        self.assertAlmostEqual(vol.end(0), 9)
        self.assertAlmostEqual(vol.start(1), 0)
        self.assertAlmostEqual(vol.end(1), 1)
        self.assertAlmostEqual(vol.start(2), 0)
        self.assertAlmostEqual(vol.end(2), 1)

        vol.reparam()
        self.assertAlmostEqual(vol.start(0), 0)
        self.assertAlmostEqual(vol.end(0), 1)
        self.assertAlmostEqual(vol.start(1), 0)
        self.assertAlmostEqual(vol.end(1), 1)
        self.assertAlmostEqual(vol.start(2), 0)
        self.assertAlmostEqual(vol.end(2), 1)

        vol.reparam((4, 10), (0, 9), (2, 7))
        vol.reparam(direction=1)
        self.assertAlmostEqual(vol.start(0), 4)
        self.assertAlmostEqual(vol.end(0), 10)
        self.assertAlmostEqual(vol.start(1), 0)
        self.assertAlmostEqual(vol.end(1), 1)
        self.assertAlmostEqual(vol.start(2), 2)
        self.assertAlmostEqual(vol.end(2), 7)

    def test_reverse(self):
        # identity mapping, control points generated from knot vector
        basis1 = BSplineBasis(4, [2, 2, 2, 2, 3, 6, 12, 12, 12, 12])
        basis2 = BSplineBasis(3, [-3, -3, -3, 20, 30, 33, 33, 33])
        basis3 = BSplineBasis(5, [0, 0, 0, 0, 0, 8, 8, 8, 8, 8])
        vol = Volume(basis1, basis2, basis3)

        u = np.linspace(2, 12, 5)
        v = np.linspace(-3, 33, 5)
        w = np.linspace(0, 8, 5)

        pt = vol(u, v, w)

        vol.reverse("v")
        pt2 = vol(u, v[::-1], w)
        self.assertAlmostEqual(np.linalg.norm(pt - pt2), 0.0)
        self.assertAlmostEqual(vol.start("v"), -3)
        self.assertAlmostEqual(vol.end("v"), 33)

        vol.reverse(2)
        pt2 = vol(u, v[::-1], w[::-1])
        self.assertAlmostEqual(np.linalg.norm(pt - pt2), 0.0)
        self.assertAlmostEqual(vol.start("w"), 0)
        self.assertAlmostEqual(vol.end("w"), 8)

    def test_faces(self):
        vol1 = Volume()
        faces = vol1.faces()
        self.assertEqual(len(faces), 6)
        # check that it all comes out in the order umin, umax, vmin, vmax, wmin, wmax
        self.assertTrue(np.allclose(faces[0][0, 0], (0, 0, 0)))
        self.assertTrue(np.allclose(faces[0][1, 1], (0, 1, 1)))
        self.assertTrue(np.allclose(faces[1][0, 0], (1, 0, 0)))
        self.assertTrue(np.allclose(faces[1][1, 1], (1, 1, 1)))
        self.assertTrue(np.allclose(faces[2][0, 0], (0, 0, 0)))
        self.assertTrue(np.allclose(faces[2][1, 1], (1, 0, 1)))
        self.assertTrue(np.allclose(faces[3][0, 0], (0, 1, 0)))
        self.assertTrue(np.allclose(faces[3][1, 1], (1, 1, 1)))
        self.assertTrue(np.allclose(faces[4][0, 0], (0, 0, 0)))
        self.assertTrue(np.allclose(faces[4][1, 1], (1, 1, 0)))
        self.assertTrue(np.allclose(faces[5][0, 0], (0, 0, 1)))
        self.assertTrue(np.allclose(faces[5][1, 1], (1, 1, 1)))

        # one parametric direction is periodic, these indices should return None
        vol2 = vf.cylinder()
        faces = vol2.faces()
        self.assertEqual(len(faces), 6)
        self.assertIsNotNone(faces[0])
        self.assertIsNotNone(faces[1])
        self.assertIsNone(faces[2])
        self.assertIsNone(faces[3])
        self.assertIsNotNone(faces[4])
        self.assertIsNotNone(faces[5])

        # two parametric directions are periodic
        vol3 = vf.torus()
        faces = vol3.faces()
        self.assertEqual(len(faces), 6)
        self.assertIsNotNone(faces[0])
        self.assertIsNotNone(faces[1])
        self.assertIsNone(faces[2])
        self.assertIsNone(faces[3])
        self.assertIsNone(faces[4])
        self.assertIsNone(faces[5])

    def test_make_identical(self):
        basis1 = BSplineBasis(4, [-1, -1, 0, 0, 1, 1, 2, 2], periodic=1)
        basis2 = BSplineBasis(3, [-1, 0, 0, 1, 1, 2], periodic=0)
        basis3 = BSplineBasis(2)
        vol1 = Volume()
        vol2 = Volume(basis1, basis2, basis3)
        vol1.refine(1)
        Volume.make_splines_identical(vol1, vol2)

        for v in (vol1, vol2):
            self.assertEqual(v.periodic(0), False)
            self.assertEqual(v.periodic(1), False)
            self.assertEqual(v.periodic(2), False)

            self.assertEqual(v.order(), (4, 3, 2))
            self.assertAlmostEqual(len(v.knots(0, with_multiplicities=True)), 11)
            self.assertAlmostEqual(len(v.knots(1, with_multiplicities=True)), 8)
            self.assertAlmostEqual(len(v.knots(2, with_multiplicities=True)), 5)

    def test_bounding_box(self):
        vol = Volume()
        bb = vol.bounding_box()
        self.assertAlmostEqual(bb[0][0], 0)
        self.assertAlmostEqual(bb[0][1], 1)
        self.assertAlmostEqual(bb[1][0], 0)
        self.assertAlmostEqual(bb[1][1], 1)
        self.assertAlmostEqual(bb[2][0], 0)
        self.assertAlmostEqual(bb[2][1], 1)

        vol.refine(2)
        vol.rotate(pi / 4, [1, 0, 0])
        vol += (1, 0, 1)
        bb = vol.bounding_box()
        self.assertAlmostEqual(bb[0][0], 1)
        self.assertAlmostEqual(bb[0][1], 2)
        self.assertAlmostEqual(bb[1][0], -sqrt(2) / 2)
        self.assertAlmostEqual(bb[1][1], sqrt(2) / 2)
        self.assertAlmostEqual(bb[2][0], 1)
        self.assertAlmostEqual(bb[2][1], 1 + sqrt(2))

    def test_controlpoint_access(self):
        v = Volume()
        v.refine(1)
        self.assertAlmostEqual(v[0, 0, 0, 0], 0)
        self.assertAlmostEqual(v[0, 0, 0][0], 0)
        self.assertAlmostEqual(v[0, 1, 0][0], 0)
        self.assertAlmostEqual(v[0, 1, 0][1], 0.5)
        self.assertAlmostEqual(v[0, 1, 0, 1], 0.5)
        self.assertAlmostEqual(v[0, 0, 2][2], 1)
        self.assertAlmostEqual(v[4][0], 0.5)
        self.assertAlmostEqual(v[4][1], 0.5)
        self.assertAlmostEqual(v[4][2], 0)
        self.assertAlmostEqual(v[13][0], 0.5)
        self.assertAlmostEqual(v[13][1], 0.5)
        self.assertAlmostEqual(v[13][2], 0.5)
        self.assertAlmostEqual(v[14][0], 1)
        self.assertAlmostEqual(v[14][1], 0.5)
        self.assertAlmostEqual(v[14][2], 0.5)

        v[0] = [0.1, 0.1, 0.1]
        v[1, 1, 1] = [0.6, 0.6, 0.6]
        v[1, 0, 0][0] = 0.4
        self.assertAlmostEqual(v[0, 0, 0][0], 0.1)
        self.assertAlmostEqual(v[0, 0, 0][1], 0.1)
        self.assertAlmostEqual(v[0, 0, 0][2], 0.1)
        self.assertAlmostEqual(v[13][0], 0.6)
        self.assertAlmostEqual(v[13][1], 0.6)
        self.assertAlmostEqual(v[13][2], 0.6)
        self.assertAlmostEqual(v[1][0], 0.4)
        self.assertAlmostEqual(v[1][1], 0)
        self.assertAlmostEqual(v[1][2], 0)

        v[:, 0, 0] = 13
        v[0, 1, 0][:] = 12
        # v[0,2,1]         = [9,8,7]
        # v[0,2,0]         = [9,8,7]
        v[0, 2, 1::-1] = [9, 8, 7]
        v[1, 2, 1::-1, :] = [[6, 5, 4], [3, 2, 1]]
        self.assertAlmostEqual(v[1, 0, 0, 0], 13)
        self.assertAlmostEqual(v[1, 0, 0, 1], 13)
        self.assertAlmostEqual(v[2, 0, 0, 2], 13)
        self.assertAlmostEqual(v[0, 1, 0, 0], 12)
        self.assertAlmostEqual(v[0, 1, 0, 2], 12)
        self.assertAlmostEqual(v[0, 2, 2, 1], 1)
        self.assertAlmostEqual(v[0, 2, 1, 0], 9)
        self.assertAlmostEqual(v[0, 2, 1, 1], 8)
        self.assertAlmostEqual(v[0, 2, 0, 2], 7)
        self.assertAlmostEqual(v[1, 2, 1, 0], 6)
        self.assertAlmostEqual(v[1, 2, 1, 1], 5)
        self.assertAlmostEqual(v[1, 2, 1, 2], 4)
        self.assertAlmostEqual(v[1, 2, 0, 0], 3)
        self.assertAlmostEqual(v[1, 2, 0, 1], 2)
        self.assertAlmostEqual(v[1, 2, 0, 2], 1)

    def test_volume(self):
        v = Volume()
        self.assertAlmostEqual(v.volume(), 1.0)
        v -= (0.5, 0.5, 0)
        v[:, :, 1, 0:2] = 0.0  # squeeze top together, creating a pyramid
        self.assertAlmostEqual(v.volume(), 1.0 / 3)

    def test_operators(self):
        v = Volume()
        v.raise_order(1, 1, 2)
        v.refine(3, 2, 1)

        # test translation operator
        v2 = v + [1, 0, 0]
        v3 = [1, 0, 0] + v
        v += [1, 0, 0]
        self.assertTrue(np.allclose(v2.controlpoints, v3.controlpoints))
        self.assertTrue(np.allclose(v.controlpoints, v3.controlpoints))

        # test scaling operator
        v2 = v * 3
        v3 = 3 * v
        v *= 3
        self.assertTrue(np.allclose(v2.controlpoints, v3.controlpoints))
        self.assertTrue(np.allclose(v.controlpoints, v3.controlpoints))

    def test_antiderivative(self):
        """Test that antiderivative inverts the derivative operation for volumes."""

        # Test 1: Trilinear volume - constant derivative in all directions
        # Volume: x(u,v,w) = u, y(u,v,w) = v, z(u,v,w) = w
        # d/du: dx/du = 1, dy/du = 0, dz/du = 0
        # Antiderivative in u should give x(u,v,w) = u^2/2, y(u,v,w) = 0, z(u,v,w) = 0
        basis1 = BSplineBasis(3, [0, 0, 0, 1, 1, 1])
        basis2 = BSplineBasis(2, [0, 0, 1, 2, 2])
        basis3 = BSplineBasis(3, [0, 0, 0, 3, 3, 3])
        # create cube [0,2]^3 with tiny variations at decimal point
        controlpoints = [
            [0.0, 0.0, 0.0], [1.1, 0.1, 0.0], [2.0, 0.0, 0.2], 
            [0.1, 1.0, 0.0], [1.2, 1.0, 0.1], [2.0, 1.0, 0.0], 
            [0.0, 2.0, 0.0], [1.0, 2.1, 0.1], [2.0, 2.0, 0.0], 

            [0.1, 0.0, 1.4], [1.1, 0.1, 1.0], [2.0, 0.3, 1.2], 
            [0.0, 1.1, 1.3], [1.1, 1.2, 1.1], [2.0, 1.0, 1.0], 
            [0.2, 2.2, 1.2], [1.2, 2.0, 1.0], [2.0, 2.0, 1.1], 

            [0.0, 0.0, 2.5], [1.1, 0.3, 2.0], [2.0, 0.0, 2.1], 
            [0.1, 1.1, 2.0], [1.1, 1.1, 2.2], [2.0, 1.1, 2.0], 
            [0.0, 2.1, 2.3], [1.3, 2.1, 2.0], [2.0, 2.0, 2.0]
        ]
        vol = Volume(basis1, basis2, basis3, controlpoints)

        # Test integration in u-direction
        integral_u = vol.get_antiderivative_volume('u')

        # Check: order increased by 1 in u-direction
        self.assertEqual(integral_u.order(0), vol.order(0) + 1)
        self.assertEqual(integral_u.order(1), vol.order(1))
        self.assertEqual(integral_u.order(2), vol.order(2))

        # Check: integral at u=0 is zero (default constant)
        u_start = integral_u.start(0)
        v_test = np.linspace(0, 1, 5)
        w_test = np.linspace(0, 1, 5)
        integral_at_start = integral_u(u_start, v_test, w_test)
        self.assertTrue(np.allclose(integral_at_start, 0.0, atol=1e-12))

        # Check: derivative of integral equals original
        u = np.linspace(0, 1, 7)
        v = np.linspace(0, 1, 7)
        w = np.linspace(0, 1, 7)
        original = vol(u, v, w)
        recovered = integral_u.derivative(u, v, w, d=(1, 0, 0))
        error = np.linalg.norm(original - recovered)
        self.assertAlmostEqual(error, 0.0, places=10)

        # Test 2: Integration in v-direction
        integral_v = vol.get_antiderivative_volume('v')

        # Check: order increased by 1 in v-direction
        self.assertEqual(integral_v.order(0), vol.order(0))
        self.assertEqual(integral_v.order(1), vol.order(1) + 1)
        self.assertEqual(integral_v.order(2), vol.order(2))

        # Check: integral at v=0 is zero
        v_start = integral_v.start(1)
        u_test = np.linspace(0, 1, 5)
        w_test = np.linspace(0, 1, 5)
        integral_at_start = integral_v(u_test, v_start, w_test)
        self.assertTrue(np.allclose(integral_at_start, 0.0, atol=1e-12))

        # Check: derivative of integral equals original
        original = vol(u, v, w)
        recovered = integral_v.derivative(u, v, w, d=(0, 1, 0))
        error = np.linalg.norm(original - recovered)
        self.assertAlmostEqual(error, 0.0, places=10)

        # Test 3: Integration in w-direction
        integral_w = vol.get_antiderivative_volume('w')

        # Check: order increased by 1 in w-direction
        self.assertEqual(integral_w.order(0), vol.order(0))
        self.assertEqual(integral_w.order(1), vol.order(1))
        self.assertEqual(integral_w.order(2), vol.order(2) + 1)

        # Check: integral at w=0 is zero
        w_start = integral_w.start(2)
        u_test = np.linspace(0, 1, 5)
        v_test = np.linspace(0, 1, 5)
        integral_at_start = integral_w(u_test, v_test, w_start)
        self.assertTrue(np.allclose(integral_at_start, 0.0, atol=1e-12))

        # Check: derivative of integral equals original
        original = vol(u, v, w)
        recovered = integral_w.derivative(u, v, w, d=(0, 0, 1))
        error = np.linalg.norm(original - recovered)
        self.assertAlmostEqual(error, 0.0, places=10)

        # Test 4: Integration with custom constant
        constant = np.array([1.0, 2.0, 3.0])
        integral_with_const = vol.get_antiderivative_volume('u', constant=constant)

        # Check: integral at start equals the constant
        u_start = integral_with_const.start(0)
        v_test = np.linspace(0, 1, 5)
        w_test = np.linspace(0, 1, 5)
        integral_at_start = integral_with_const(u_start, v_test, w_test)
        for i in range(len(v_test)):
            for j in range(len(w_test)):
                error = np.linalg.norm(integral_at_start[0, i, j, :] - constant)
                self.assertLess(error, 1e-12)

        # Check: derivative of integral equals original
        u = np.linspace(0, 1, 9)
        v = np.linspace(0, 1, 9)
        w = np.linspace(0, 1, 9)
        original = vol(u, v, w)
        recovered = integral_with_const.derivative(u, v, w, d=(1, 0, 0))
        error = np.linalg.norm(original - recovered)
        self.assertAlmostEqual(error, 0.0, places=9)

        # Test 5: Verify double integration (integrate in u, then in v)
        integral_u = vol.get_antiderivative_volume('u')
        integral_uv = integral_u.get_antiderivative_volume('v')

        # Check: orders increased in both directions
        self.assertEqual(integral_uv.order(0), vol.order(0) + 1)
        self.assertEqual(integral_uv.order(1), vol.order(1) + 1)
        self.assertEqual(integral_uv.order(2), vol.order(2))

        # Check: mixed derivative d^2/dudv equals original
        u = np.linspace(0, 1, 7)
        v = np.linspace(0, 1, 7)
        w = np.linspace(0, 1, 7)
        original = vol(u, v, w)
        recovered = integral_uv.derivative(u, v, w, d=(1, 1, 0))
        error = np.linalg.norm(original - recovered)
        self.assertAlmostEqual(error, 0.0, places=9)

        # Test 6: Verify triple integration (u, then v, then w)
        integral_uvw = integral_uv.get_antiderivative_volume('w')

        # Check: orders increased in all directions
        self.assertEqual(integral_uvw.order(0), vol.order(0) + 1)
        self.assertEqual(integral_uvw.order(1), vol.order(1) + 1)
        self.assertEqual(integral_uvw.order(2), vol.order(2) + 1)

        # Check: mixed derivative d^3/dudvdw equals original
        original = vol(u, v, w)
        recovered = integral_uvw.derivative(u, v, w, d=(1, 1, 1))
        error = np.linalg.norm(original - recovered)
        self.assertAlmostEqual(error, 0.0, places=8)

        # Test 7: Rational volumes should raise an error
        cp_rational = [
            [0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 1],
            [0, 0, 1, 1], [1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]
        ]
        basis = BSplineBasis(2, [0,0,1,1])
        vol_rational = Volume(basis, basis, basis, cp_rational, rational=True)

        with self.assertRaises(RuntimeError):
            vol_rational.get_antiderivative_volume('u')

        # Test 8: Direction parameter variants (numeric and string)
        integral_0 = vol.get_antiderivative_volume(0)
        integral_u = vol.get_antiderivative_volume('u')
        self.assertTrue(np.allclose(integral_0.controlpoints, integral_u.controlpoints))

        integral_1 = vol.get_antiderivative_volume(1)
        integral_v = vol.get_antiderivative_volume('v')
        self.assertTrue(np.allclose(integral_1.controlpoints, integral_v.controlpoints))

        integral_2 = vol.get_antiderivative_volume(2)
        integral_w = vol.get_antiderivative_volume('w')
        self.assertTrue(np.allclose(integral_2.controlpoints, integral_w.controlpoints))

        # Test 9: Invalid direction should raise an error
        with self.assertRaises(ValueError):
            vol.get_antiderivative_volume(3)
        with self.assertRaises(ValueError):
            vol.get_antiderivative_volume('x')


if __name__ == "__main__":
    unittest.main()
