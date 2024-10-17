import numpy as np

def estimate_mot_pwr(speed,                 # v(t): [m/s]
                     accel,                 # a(t): [m/s²]
                     alt,                   # z(t): [m]
                     road_grad,             # alpha(t): [%]
                     amb_temp,              # T(t): [°C]
                     weight,                # m(t): [kg]
                     c_w_a,                 # [m²]
                     roll_coeff,            # [-]
                     rot_inertia = 0,       # > 0 , default = 0
                     eta_mech = 1,          # 0 <= 1, default = 1   # Drivetrain Efficiency Factor (e.g. gearbox, mechanical losses)
                     eta_mot = 1,           # 0 <= 1, default = 1   # Motor Efficiency Factor (mechanical <-> electrical)
                     ):
    
    ################################################################################################
    # constants:
    g_amb = 9.81    # [m/s²]

    # air_dens = p / (R * T) = 101325 x (1.0 − Z x 0.0000225577)**5.2559 / (287.05 J/kg-K * T)  
    air_dens = (101325*(1.0 - alt * 2.25577*(10**-5) )**5.2559) / (287.05*(273.15 + amb_temp )) # [kg/m³]
    # limit value:
    air_dens = np.clip(air_dens, 1.1, 1.3)

    # convert road slope to rad:
    alpha_rad = np.arctan(road_grad/100)

    ################################################################################################
    '''
    For the Acceleration Force, the rotational inertia of the rotating vehicle parts should 
    be considered. However the exact calculation of the inertias of the truck with respect to 
    the drive shaft is extremely elaborate. This can be resembled by an additional inertia 
    part that contributes to the "equivalent vehicle mass". The value of 1.05 ... 1.1 was 
    introduced based on the literatures of IC Engine trucks. The value still needs to be 
    validated for electric vehicle:
    '''
    m = weight                      # vehicle mass [kg]
    m_e = m * (1 + rot_inertia)     # equivalent vehicle mass [kg]

    ################################################################################################
    # Now the resistance forces that act on the vehicle wheels can be calculated over time:

    # Momentary Vehicle Tractive Force [N]:
    # F_res =       F_Air             0.5 * air_dens * c_w_a * v²           
    #             + F_Roll            m * g_amb * r_coeff * cos(alpha)     
    #             + F_Gradient        m * g_amb * sin(alpha)
    #             + F_Accel           (m + J_red/(r_dyn²)) * a = m_e * a
    # F_res = m_e * a + m*g_amb*(sin(alpha) + r_coeff*cos(alpha))  + air_dens/2 * c_w_a * v² 
    ################################################################################################

    # total resistance force: [N]
    F_res = m_e * accel  +  m * g_amb * (np.sin(alpha_rad) + roll_coeff * np.cos(alpha_rad))  +  air_dens/2 * c_w_a * speed**2

    # this results in the required tractive power at the wheels: [kW]
    P_mech = F_res * speed / 1000    #[kW]

    ################################################################################################
    # apply efficiency factors of drivetrain and Emotors:
    eta = eta_mech * eta_mot

    # Note: consider different eta operation for power output/input! --> required output is higher, possible recuperation is lower
    eta_total = (P_mech < 0) * 1/eta + (P_mech >= 0) * eta

    # electrical Power of Motor [kW]:
    P_el = P_mech / eta_total

    ################################################################################################
    '''
    SERVICE BRAKE LOSS
    apply additional factor for recuperational sections, to represent non-ideal driver behaviour, as the driver 
    influences the amount of recuperation braking, resulting in loss of braking energy (service brake loss). This is in order to flatten
    the negative peaks, that are theoretically calculated if the driver is braking with the Emotor only.
    
    # this leads to following recup efficiencies:
    
     | P_mech [kW]   | recup_eff [%]   |
     |---------------|-----------------|
     | 0 ... -250    | 100 %           |
     | -250 ... -350 | 90 %            |
     | -350 ... -450 | 75 %            |
     | < -450        | 50 %            |
    '''
    recup_eff = 1 - ((P_mech < -250) * 0.1) - ((P_mech < -350) * 0.15)- ((P_mech < -450) * 0.25)

    # deactivate:
    P_el = recup_eff * P_el
    
    ################################################################################################
    return P_el