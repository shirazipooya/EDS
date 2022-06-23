# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

from typing import Dict, Optional, Tuple, Union, List

import numpy as np
import pandas as pd


class Parameters():
    
    def __init__(
        self,
        # Spray
        spray_number: List[int] = [1, 2, 3],
        spray_moment: List[int] = [30, 45, 60],
        spray_eff: List[float] = [0.5, 0.5, 0.5],
        # Genetic Mechanistic
        p_opt: List[float] = [7, 10, 14],
        rc_opt_par: List[float] = [0.35, 0.25, 0.15],
        rrlex_par: List[float] = [0.1, 0.01, .0001],
        # Crop Parameters From File
        crop_mechanistic: List[str] = None,
        crop_parameters_path: List[str] = None,
    ) -> None:
        self.spray_number = spray_number
        self.spray_moment = spray_moment
        self.spray_eff = spray_eff
        
        self.p_opt = p_opt
        self.rc_opt_par = rc_opt_par
        self.rrlex_par = rrlex_par
        
        self.crop_mechanistic = crop_mechanistic
        self.crop_parameters_path = crop_parameters_path
    
    
    def spray(
        self
    ) -> pd.DataFrame:
        
        spray_para = pd.DataFrame(
            {
                "spray_number": self.spray_number,
                "spray_moment": self.spray_moment,
                "spray_eff": self.spray_eff
            }
        )
        
        return spray_para
    
    
    def genetic_mechanistic(
        self
    ) -> Dict:
        
        genetic_mechanistic_para = {
            
            "Susceptible" : {
                "p_opt" : self.p_opt[0],
                "rc_opt_par" : self.rc_opt_par[0],
                "rrlex_par" : self.rrlex_par[0]
            },
            
            "Moderate" : {
                "p_opt" : self.p_opt[1],
                "rc_opt_par" : self.rc_opt_par[1],
                "rrlex_par" : self.rrlex_par[1]
            },
            
            "Resistant" : {
                "p_opt" : self.p_opt[2],
                "rc_opt_par" : self.rc_opt_par[2],
                "rrlex_par" : self.rrlex_par[2]
            }
            
        }
        
        return genetic_mechanistic_para
    
    
    def crop(
        self
    ) -> Dict:
        
        crop_para = {
            "Corn" : {
                
                'ip_t_cof': pd.DataFrame(
                    {
                        0: [10.00, 13.00, 15.50, 17.00, 20.00, 26.00, 30.00, 35.00],
                        1: [0.00, 0.14, 0.27, 0.82, 1.00, 0.92, 0.41, 0.00]
                    }
                ),
                
                'p_t_cof': pd.DataFrame(
                    {
                        0: [15.00, 20.00, 25.00],
                        1: [0.60, 0.81, 1.00]
                    }
                ),
                
                'rc_t_input': pd.DataFrame(
                    {
                        0: [15.00, 20.00, 22.50, 24.00, 26.00, 30.00],
                        1: [0.22, 1.00, 0.44, 0.43, 0.41, 0.22]
                    }
                ),
                
                'dvs_8_input': pd.DataFrame(
                    {
                        0: [110.00, 200.00, 350.00, 475.00, 610.00, 740.00, 1135.00, 1660.00, 1925.00, 2320.00, 2700.00],
                        1: [0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 8.00, 9.00, 10.00, 11.00]
                    }
                ),
                
                'rc_a_input': pd.DataFrame(
                    {
                        0: [0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00],
                        1: [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
                    }
                ),
                
                'fungicide': pd.DataFrame(
                    {
                        0: [1, 2, 3],
                        1: [45, 62, 79],
                        2: [0.4, 0.4, 0.4]
                    }
                ),
                
                'fungicide_residual': pd.DataFrame(
                    {
                        0: [0.00, 5.00, 10.00, 15.00, 20.00],
                        1: [1.00, 0.80, 0.50, 0.25, 0.00]
                    }
                )        
            },
            
            "Soy" : {
                
                'ip_t_cof': pd.DataFrame(
                    {
                        0: [5.00, 11.00, 17.00, 23.00, 29.00, 35.00],
                        1: [0.00, 0.33, 0.60, 1.00, 1.00, 0.00]
                    }
                ),
                
                'p_t_cof': pd.DataFrame(
                    {
                        0: [0.00, 4.00, 8.00, 12.00, 16.00, 20.00, 24.00, 28.00, 32.00, 36.00, 40.00],
                        1: [0.00, 0.31, 0.39, 0.53, 0.82, 1.00, 1.00, 0.75, 0.60, 0.30, 0.00]
                    }
                ),
                
                'rc_t_input': pd.DataFrame(
                    {
                        0: [10.00, 12.50, 15.00, 17.50, 20.00, 23.00, 25.00, 27.50, 30.00],
                        1: [0.00, 0.57, 0.88, 1.00, 1.00, 0.86, 0.61, 0.41, 0.00]
                    }
                ),
                
                'dvs_8_input': pd.DataFrame(
                    {
                        0: [137.00, 366.00, 595.00, 824.00, 1053.00, 1282.00, 1511.00, 1740.00],
                        1: [0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00]
                    }
                ),
                
                'rc_a_input': pd.DataFrame(
                    {
                        0: [0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00],
                        1: [1.0000, 1.0000, 1.0000, 0.1000, 0.0001, 0.0001, 0.0001, 0.0001]
                    }
                ),
                
                'fungicide': pd.DataFrame(
                    {
                        0: [1, 2, 3],
                        1: [45, 62, 79],
                        2: [0.4, 0.4, 0.4]
                    }
                ),
                
                'fungicide_residual': pd.DataFrame(
                    {
                        0: [0.00, 5.00, 10.00, 15.00, 20.00],
                        1: [1.00, 0.80, 0.50, 0.25, 0.00]
                    }
                )        
            }  
        }
        
        return crop_para
    
    
    def corp_from_file(
        self
    ) -> Dict:
        
        para = ["ip_t_cof", "p_t_cof", "rc_t_input", "dvs_8_input", "rc_a_input", "fungicide", "fungicide_residual"]
        
        crop_para = dict()
        
        if (self.crop_parameters_path is not None) and (self.crop_mechanistic is not None):
            if len(self.crop_mechanistic) == len(self.crop_parameters_path):
                for i in range(len(self.crop_mechanistic)):
                    crop_para[self.crop_mechanistic[i]] = dict()
                    for p in para:
                        crop_para[self.crop_mechanistic[i]][p] = pd.read_excel(
                            self.crop_parameters_path[i],
                            engine="openpyxl",
                            sheet_name=p,
                            header=None
                        )
        return crop_para