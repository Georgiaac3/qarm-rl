class QArmKinematics:
    def get_jacobian(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calcule le jacobien (J) de la cinématique directe du robot.
        - q : vecteur colonne des angles articulaires de la dynamique géométrique (4,1)
        - retourne : jacobien (3,4)
        - les paramètres géométriques du robot sont définis dans core/config.py
        """

        if q.shape != (4, 1):
            raise ValueError("q doit être un vecteur colonne de dimension (4, 1)")

        c1, s1, c2, s2, c3, s3, c23, s23 = get_trig_values(q)

        J = np.array(
            [
                [-l2 * s1 * c2 + l3 * s1 * s23, -l2 * c1 * s2 - l3 * c1 * c23, -l3 * c1 * c23, 0],
                [l2 * c1 * c2 - l3 * c1 * s23, -l2 * s1 * s2 - l3 * s1 * c23, -l3 * s1 * c23, 0],
                [0, -l2 * c2 + l3 * s23, l3 * s23, 0],
            ]
        )

        return J

    def get_djacobian(self, q: NDArray[np.float64], dq: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calcule la dérivée du jacobien (dJ) de la cinématique directe du robot.
        - q : vecteur colonne des angles articulaires de la dynamique géométrique (4,1)
        - dq : vecteur colonne des vitesses articulaires de la dynamique géométrique (4,1)
        - retourne : dérivée du jacobien (3,4)
        - les paramètres géométriques du robot sont définis dans core/config.py
        """

        if q.shape != (4, 1) or dq.shape != (4, 1):
            raise ValueError("q et dq doivent être des vecteurs colonne de dimension (4, 1)")

        dq1 = dq[0, 0]
        dq2 = dq[1, 0]
        dq3 = dq[2, 0]

        c1, s1, c2, s2, c3, s3, c23, s23 = get_trig_values(q)

        dJ = np.array(
            [
                [
                    (-l2 * c1 * c2 + l3 * c1 * s23) * dq1
                    + (l2 * s1 * s2 + l3 * s1 * c23) * dq2
                    + (l3 * s1 * c23) * dq3,
                    (l2 * s1 * s2 + l3 * s1 * c23) * dq1
                    + (-l2 * c1 * c2 + l3 * c1 * s23) * dq2
                    + (l3 * c1 * s23) * dq3,
                    l3 * s1 * c23 * dq1 + (l3 * c1 * s23) * dq2 + (l3 * c1 * s23) * dq3,
                    0,
                ],
                [
                    (-l2 * s1 * c2 + l3 * s1 * s23) * dq1
                    + (-l2 * c1 * s2 - l3 * c1 * c23) * dq2
                    + (-l3 * c1 * c23) * dq3,
                    (-l2 * c1 * s2 - l3 * c1 * c23) * dq1
                    + (-l2 * s1 * c2 + l3 * s1 * s23) * dq2
                    + (l3 * s1 * s23) * dq3,
                    -l3 * c1 * c23 * dq1 + (l3 * s1 * s23) * dq2 + (l3 * s1 * s23) * dq3,
                    0,
                ],
                [
                    0,
                    (l2 * s2 + l3 * c23) * dq2 + (l3 * c23) * dq3,
                    (l3 * c23) * dq2 + (l3 * c23) * dq3,
                    0,
                ],
            ]
        )

        return dJ

    def forward_kinematics(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calcule la position cartésienne de l'effecteur en fonction des angles articulaires q.
        - q : vecteur colonne des angles articulaires de la dynamique géométrique (4,1)
        - retourne : position cartésienne de l'effecteur (3,1)
        - les paramètres géométriques du robot sont définis dans core/config.py
        """
        if q.shape != (4, 1):
            raise ValueError("q doit être un vecteur colonne de dimension (4, 1)")

        c1, s1, c2, s2, c3, s3, c23, s23 = get_trig_values(q)

        x = c1 * (l2 * c2 - l3 * s23)
        y = s1 * (l2 * c2 - l3 * s23)
        z = l1 - l2 * s2 - l3 * c23
        return np.array([[x], [y], [z]])
