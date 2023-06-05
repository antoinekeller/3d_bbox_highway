#!/usr/bin/env python3

"""
Implement Kalman Filter class.
The variables we want to estimate are x and y positions, and bbox dimensions. 
For this we assume, a constant velocity regarding y and a constant position regarding x
(we assume that cars are not changing lane).
We assume constant width, height and length.

Our 8 state variables:
- x
- y
- speed along y (vy)
- width
- height
- length

Our 5 measurements:
- x
- y
- width
- height
- length
"""

import numpy as np

TIME_INTERVAL = 1.0 / 30.0


class KalmanFilter:
    """
    Kalman Filter with CV model
    Define state, covariance, predict & update steps
    """

    def __init__(self, nx, nz, first_measurement):
        self.nx = nx  # x, y, vy, w, h, l
        self.nz = nz  # x, y, w, h, l
        self.K = np.zeros((nx, nz))
        assert first_measurement.shape == (nz,)

        self.estimate = np.array(
            [
                first_measurement[0],  # x
                first_measurement[1],  # y
                0,  # vy
                first_measurement[2],  # w
                first_measurement[3],  # h
                first_measurement[4],  # l
            ]
        ).reshape(nx, 1)

        # Initial covariance estimate
        self.P = np.diag(
            [
                100**2,
                100**2,
                500**2,
                5**2,
                5**2,
                5**2,
            ]
        )
        assert np.all(self.P.T == self.P)

        # Jacobian measurement matrix
        self.H = np.zeros((nz, nx))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 3] = 1
        self.H[3, 4] = 1
        self.H[4, 5] = 1

        # Measurement noise R
        XY_STD_MEASUREMENT = 5  # 5 pixels accuracy for x/y position
        WHL_STD_MEASUREMENT = 10  # 10 pixels std for width/height
        self.R = np.diag(
            [
                XY_STD_MEASUREMENT**2,
                XY_STD_MEASUREMENT**2,
                WHL_STD_MEASUREMENT**2,
                WHL_STD_MEASUREMENT**2,
                WHL_STD_MEASUREMENT**2,
            ]
        )  # measurement uncertainty

        self.I = np.identity(nx)

        # Process noise (the most touchy part)
        # values too low --> lag error
        # values too high --> KF follows measurements and we have noisy estimation
        # It s possible that you change these values for your current situation
        self.Q = np.zeros((nx, nx))
        MAX_SPEED = 5
        MAX_TR = 3
        self.Q[0, 0] = (TIME_INTERVAL * MAX_SPEED) ** 2
        self.Q[1, 1] = (TIME_INTERVAL * MAX_SPEED) ** 2
        self.Q[2, 2] = MAX_SPEED**2
        self.Q[3, 3] = 1**2
        self.Q[4, 4] = 1**2
        self.Q[5, 5] = 1**2

        # Model matrix (constant velocity and turn rate)
        self.F = np.zeros((nx, nx))
        self.F[0, 0] = 1
        self.F[1, 1] = 1
        self.F[2, 2] = 1
        self.F[1, 2] = TIME_INTERVAL
        self.F[3, 3] = 1
        self.F[4, 4] = 1
        self.F[5, 5] = 1

        assert np.all(self.Q.T == self.Q)

        assert self.estimate.shape == (nx, 1)
        assert self.P.shape == (nx, nx)
        assert self.H.shape == (nz, nx)
        assert self.I.shape == (nx, nx)
        assert self.R.shape == (nz, nz)
        assert self.Q.shape == (nx, nx)
        assert self.K.shape == (nx, nz)

    def predict(self):
        """
        Predict state and covariance (linear)
        """

        self.estimate = self.F.dot(self.estimate)  # no control on the system

        # Predict covariance
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q

    def update(self, measurement):
        """
        Update step
        Update Kalman gain K
        Update state estimate
        Update state convariance P
        """
        # Update Kalman Gain

        assert measurement.shape == (self.nz,)

        # Linearization around current estimate
        estimate = self.estimate.reshape(self.nx, 1)

        # Innovation
        S = self.H.dot(self.P).dot(self.H.T) + self.R

        # Kalman gain
        self.K = self.P.dot(self.H.T).dot(np.linalg.inv(S))

        # assert that this was invertible

        assert self.estimate.shape == (self.nx, 1)
        assert self.K.shape == (self.nx, self.nz)

        # State update equation
        diff = measurement.reshape(self.nz, 1) - self.H.dot(estimate)
        self.estimate = estimate + self.K.dot(diff)

        assert self.estimate.shape == (self.nx, 1)

        # Covariance update equation
        self.P = (self.I - self.K.dot(self.H)).dot(self.P)
