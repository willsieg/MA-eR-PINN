def CreateVehicle(name):

    class Vehicle:
        def __init__(self, name):
            self.name = name
            self.prm = {}

        def __str__(self):
            return f"{self.name}"
    
        def update(self, parameter, value):
            self.prm.update({parameter: value})

    prm_list_base = {       # APPLIES TO ALL VARIANTS!
        "cooling_body": 1,  # yes/no
        "num_driven_axles": 1,  # int
        "num_axles": 3, # int
        "base_weight_kg": 10000,    # [kg]
        "tire_radius": 0.492,   # [m]
        "gear_ratio": 15.1, # [-]
        "speed_limit_cpc": 89,  # [kmph]
        "speed_limit_tol": 10,  # [kmph]
        "num_emotors": 2, # int 
        "srv_brk_energy_fac": 0.02, # [-]
        "veh_height": 4.00, # [m]
        "veh_width": 2.55,  # [m]
        "hv_batt_physical_up_lmt": 97,  # [%]
        "hv_batt_physical_low_lmt": 5,  # [%]
        }
    
    # allocate vehicles to six different groups. Inside a group, all parameters are the same:
    V_Groups = [[1,2,12],               # V1,V2,V12
                [4,11,13,14,16,17],     # V4, V11, V13, V14, V16, V17
                [10],                   # V10
                [15,19],                # V15, V19
                [18],                   # V18
                [101,102]               # V101, V102
                ]

    add_keys =  ["veh_prm","num_axles","base_weight_kg","max_weight_kg","tire_roll_res_coeff","c_w_a","num_batt_packs",
                 "hv_batt_total_capacity","hv_batt_useable_capacity","veh_length","hv_batt_installed_capacity"]

    add_values = [
        [11,	2,	8400,	19000,	0.00550,	4.9600,	3,	325.60,	309.120,	10.000,	336],
        [13,	3,	10000,	27000,	0.00550,	4.9600,	3,	325.60,	309.120,	10.000,	336],
        [14,	3,	12500,	40000,	0.00650,	8.8260,	3,	325.60,	309.120,	18.750,	336],
        [15,	3,	10400,	27000,	0.00550,	4.9600,	4,	434.00,	411.240,	10.000,	447],
        [12,	3,	9400,	27000,	0.00550,	4.9600,	3,	325.60,	309.120,	10.000,	336],
        [16,	5,	8600,	40000,	0.00450,	6.3500,	3,	325.60,	309.120,	16.500,	336]
        ]


    id = int(name)
    idx = [V_Groups.index(g) for g in V_Groups if id in g]
    prm_list_variant = dict(zip(add_keys, add_values[idx[0]]))
    prm_list_variant = {**prm_list_variant,**prm_list_base}

    V = Vehicle(name)
    for p in prm_list_variant.keys():
        V.update(p, prm_list_variant[p])

    return V

    '''
    # Vehicles: Example for V14
    prm_list = {   
        "veh_prm": 13,  # int
        "cooling_body": 1,  # yes/no
        "num_driven_axles": 1,  # int
        "num_axles": 3, # int
        "base_weight_kg": 10000,    # [kg]
        "max_weight_kg": 27000, # [kg]
        "tire_roll_res_coeff": 0.0055, # [-]
        "tire_radius": 0.492,   # [m]
        "c_w_a": 4.96,  # [m**2]
        "gear_ratio": 15.1, # [-]
        "num_batt_packs": 3, # int
        "hv_batt_total_capacity": 325.60, # [kWh]
        "hv_batt_useable_capacity": 309.12, # [kWh]
        "speed_limit_cpc": 89,  # [kmph]
        "speed_limit_tol": 10,  # [kmph]
        "num_emotors": 2, # int 
        "srv_brk_energy_fac": 0.02, # [-]
        "veh_height": 4.00, # [m]
        "veh_width": 2.55,  # [m]
        "veh_length": 9.30, # [m]
        "hv_batt_physical_up_lmt": 97,  # [%]
        "hv_batt_physical_low_lmt": 5,  # [%]
        "hv_batt_installed_capacity": 336.00    # [kWh]
        }'''

