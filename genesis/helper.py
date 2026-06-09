import numpy as np

from dynamics import transform_angles, Valim, ktGR, R, kvGR

def get_tau_from_pwm(pwm, dphi_mes):
    """
    pwm : the pwm command, np.array column vector
    dphi_mes : mesure angles from the motors, np.array column vector
    Return the torque (tau) to apply to the motors from the pwm command
    """
    return pwm
    _, dq_mes, _ = transform_angles(np.zeros(4).reshape(-1, 1), dphi_mes, np.zeros(4).reshape(-1, 1))

    #print("dq_mes:", dq_mes.flatten())
    Vcmd = pwm*Valim#*10

    tau = (Vcmd) * (ktGR/R)
    #tau = (Vcmd - kvGR*dq_mes) * (ktGR/R)

    return tau