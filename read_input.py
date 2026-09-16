from typing import Dict, Tuple
import numpy as np
import yaml

def read_input() -> Tuple[Dict, Dict]:
    """Read vessel parameters and hydrodynamic data from input files.
    
    Reads vessel parameters from 'input.yml' and hydrodynamic coefficients
    from the file specified in input.yml.
    
    Returns:
        Tuple containing:
            vessel_params (Dict): Basic vessel parameters including mass, dimensions
            hydrodynamic_data (Dict): Hydrodynamic coefficients and related data
    """
    with open('/workspaces/mavlab/ros2_ws/src/student/tut_03/input.yml', 'r') as file:
        input_data = yaml.safe_load(file)

    # Read hydrodynamic coefficients file
    hydrodynamic_file = input_data.pop('hydrodynamic_coefficients')
    with open(hydrodynamic_file, 'r') as file:
        hydrodynamic_data = yaml.safe_load(file)

    # Extract base vessel parameters
    vessel_params = {
        'rho': input_data['density'],
        'g': input_data['gravity'],
        'L': input_data['length'],
        'U': input_data['speed'],
        'mass': input_data['mass'],
        'cog': np.array(input_data['cog']),
        'gyration': np.array(input_data['gyration']),
        'T_delta': input_data['T_delta'],
        'initial_velocity': np.array(input_data['initial_velocity']),
        'initial_position': np.array(input_data['initial_position']),
        'sim_time': input_data['sim_time'],
        'time_step': input_data['time_step'],
        'control_type': input_data['control_type']
    }

    return vessel_params, hydrodynamic_data