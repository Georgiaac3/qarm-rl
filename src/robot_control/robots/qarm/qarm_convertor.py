class QArmConvertor:
    def convert_trig_values(self, c1, s1, c2, s2, c3, s3, c23, s23) -> list:
        """
        Convertit des valeurs trigonométriques de la convention old vers la convention new.

        Retourne:
            Une liste dans l'ordre (c1_new, s1_new, c2_new, s2_new, c3_new, s3_new, c23_new, s23_new).
        """
        # old -> new:
        # c2_old = s2_new, s2_old = -c2_new
        # c23_old = s23_new, s23_old = -c23_new
        return [c1, s1, s2, -c2, c3, s3, s23, -c23]
