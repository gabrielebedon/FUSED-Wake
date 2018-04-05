import numpy as np
import os
import shutil
import subprocess
from scipy.interpolate import interp1d

class FFL(object):
    # The different versions and their respective inputs
    inputs = {
        'ffl': ['WF', 'WS', 'WD', 'TI'],
    }
    # Default variables for running the wind farm flow model
    defaults = {
        'ncpus': 4,
        'rho': 1.225,
        'version': 'ffl',
        'fflpath': 'C:\\Users\\bedon\\Documents\\FarmFlow 3.0b\\',
        'fflproj': 'C:\\Users\\bedon\\Documents\\FarmFlow 3.0b\\Projects\\TopFarm\\',
    }
    def __init__(self, **kwargs):
        self.set(self.defaults)
        self.set(kwargs)

    @property
    def versions(self):
        versions = list(self.inputs.keys())
        versions.sort()
        return versions

    def set(self, dic):
        """ Set the attributes of a dictionary as instance variables. Prepares
        for the different versions of the wake model

        Parameters
        ----------
        dic: dict
            An input dictionary
        """
        for k, v in dic.items():
            setattr(self, k, v)

        # Preparing for the inputs for the fortran version
        if 'WF' in dic:
            self.x_g, self.y_g, self.z_g = self.WF.get_T2T_gl_coord2()
            self.dt = self.WF.rotor_diameter
            self.p_c = self.WF.power_curve
            self.ct_c = self.WF.c_t_curve
            self.ws_ci = self.WF.cut_in_wind_speed
            self.ws_co = self.WF.cut_out_wind_speed
            self.ct_idle = self.WF.c_t_idle

    def _get_kwargs(self, version):
        """Prepare a dictionary of inputs to be passed to the wind farm flow model

        Parameters
        ----------
        version: str
            The version of the wind farm flow model to run ['py0' | 'py1' | 'fort0']
        """
        if 'py' in version:
            return {k:getattr(self, k) for k in self.inputs[version] if hasattr(self, k)}
        if 'fort' in version:
            # fortran only get lowercase inputs
            return {(k).lower():getattr(self, k) for k in self.inputs[version] if hasattr(self, k)}

    def ffl(self):
        # Refresh folder
        if os.path.exists(self.fflproj):
            shutil.rmtree(self.fflproj)
        os.makedirs(self.fflproj)
        # reduce number of wind turbines
        power_curve_un,index_un,index_rev = np.unique(self.WF.power_curve,axis=0,return_index=True,return_inverse=True)
        c_t_curve_un = [self.WF.c_t_curve[i] for i in index_un]
        hub_height_un = [self.WF.hub_height[i] for i in index_un]
        rotor_diameter_un = [self.WF.rotor_diameter[i] for i in index_un]
        n_turbines_un = len(power_curve_un)
        # turbines file
        turbinesFile = open(self.fflproj + 'turbines','w')
        for l_turb in range(n_turbines_un):
            turbinesFile.write(f'TopFarmTurb{l_turb:.0f}.trb {hub_height_un[l_turb]:.1f}\n')
        for l_turb in range(10-n_turbines_un):    
            turbinesFile.write(f'TopFarmTurb{0:.0f}.trb {hub_height_un[0]:.1f}\n')
        turbinesFile.close()
        # trb files
        for l_turb in range(n_turbines_un):
            trbFile = open(self.fflpath + f'Turbines\\TopFarmTurb{l_turb:.0f}.trb','w+')
            trbFile.write(f'[Type]\nTopFarm Turbine {l_turb:.0f}\n\n')
            trbFile.write(f'[RotorDiameter]\n{rotor_diameter_un[l_turb]:.1f} [m]\n\n')
            trbFile.write(f'[HubHeight]\n{hub_height_un[l_turb]:.1f} [m]\n\n')
            trbFile.write(f'[AirDensity]\n{self.rho:.3f} [kg/m3]\n\n')
            trbFile.write(f'[Performance]\nStandard\nU[m/s] P[kW] CT[-]\n')
            for k_ws in range(len(power_curve_un[l_turb])):
                trbFile.write(f'{power_curve_un[l_turb][k_ws][0]:.1f} {power_curve_un[l_turb][k_ws][1]:.1f} {c_t_curve_un[l_turb][k_ws][1]:.3f}\n')
            trbFile.close()
        # farm file
        n_turbines = len(self.WF.hub_height)
        farmFile = open(self.fflproj + 'farm','w')
        for l_x in range(n_turbines):
            farmFile.write(f'{index_rev[l_x]+1:.0f} 0 {self.WF.position[l_x][0]:.1f} {self.WF.position[l_x][1]:.1f} 0.0 WTG_{l_x:.0f} !\n')
        farmFile.close()
        # input file
        inputFile = open(self.fflproj + 'input','w')
        inputFile.write(f'project_dir {self.fflproj:s}\n')
        inputFile.write(f'output_dir {self.fflproj:s}output\n')
        inputFile.write(f'reference_height {self.WF.WT.hub_height:.1f}\n')
        inputFile.write(f'air_density {self.rho:.3f}\n')
        inputFile.write(f'max_cpu {self.ncpus:.0f}\n')
        inputFile.write(f'wind_option 1\n')
        for k_ws in range(self.WS.shape[0]):
            inputFile.write(f'wind_data {self.WD[k_ws]:.0f} {self.WS[k_ws]:.1f} {self.TI[k_ws]:.3f}\n')
        inputFile.close()

        # Run the executable
        subprocess.call([self.fflpath+'Source\\FarmFlow.exe',self.fflproj+'input'])
        
        # read Results.txt
        self.u_wt = np.zeros([self.WS.shape[0], n_turbines])
        self.p_wt = np.zeros([self.WS.shape[0], n_turbines])
        with open(self.fflproj + '\\output\\Results.txt') as f:
            for line in f:
                data = line.split()
                if len(data) == 24 and data[0].isdigit():
                    k_ws = np.where(np.round(self.WS,decimals=2)==float(data[3]))
                    l_wd = np.where(np.round(self.WD,decimals=1)==float(data[2]))
                    m_pos = np.intersect1d(k_ws,l_wd)
                    i_wt = int(data[1].split('_')[1])
                    self.u_wt[m_pos, i_wt] = data[8]
                    self.p_wt[m_pos, i_wt] = data[15]
        self.p_wt *= 1.0E3
        # calculating thrust coefficients
        self.c_t = np.zeros([self.WS.shape[0], n_turbines])
        for l_turb in range(n_turbines_un):
            f_c_t = interp1d([subl[0] for subl in c_t_curve_un[l_turb]],[subl[1] for subl in c_t_curve_un[l_turb]])
            id_turb = np.where(index_rev == l_turb)
            self.c_t[:,id_turb] = f_c_t(self.u_wt[:,id_turb])

        if len(self.WS) == 1: # We are only returning a 1D array
            self.p_wt = self.p_wt[0]
            self.u_wt = self.u_wt[0]
            self.c_t = self.c_t[0]

    def __call__(self, **kwargs):
        self.set(kwargs)
        if hasattr(self, 'version'):
            getattr(self, self.version)()
            if not self.version in self.versions:
                raise Exception("Version %s is not valid: version=[%s]"%(self.version, '|'.join(self.versions)))
        else:
            raise Exception("Version hasn't been set: version=[%s]"%('|'.join(self.versions)))
        return self
