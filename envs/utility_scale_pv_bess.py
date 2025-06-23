import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import copy
import csv
import os

# local modules
import helper_fns as hlp
import config


class UtilityScalePVBESS(gym.Env):
  metadata = {"render_modes": ["evaluation"], "render_fps": 15}

  def __init__(self, 
              forecast_scada_timeseries: np.ndarray,
              bess_properties: dict,
              sec_per_step: float,
              soc_levels: int,
              render_mode: str = None,
              csv_path: str = None,
              ):
    super(UtilityScalePVBESS, self).__init__()
    self.render_mode = render_mode
    self._evaluation_flag = False

    self.previous_state = None
    self.state = None
    self.sec_per_step = sec_per_step

    # note: each daytime forecast and scada record is naturally padded with zero values at night
    self.data = forecast_scada_timeseries
    self.index = None # indexing the data array in the day*time dimension
    # note: np.nonzero() returns a tuple of arrays, so [0] is used to get the only array from the tuple.
    self.nonzero_indices = np.nonzero(self.data[:,-1])[0] # simulation will start from any Pm > 0 states
    self.cluster_head_indices = hlp.beginning_of_nonzero_cluster_indices(self.nonzero_indices) # indexing the beginning of each day

    self.max_steps = None
    self.last_cluster = False

    # assign bess_properties to constants used by _bess_dynamics()
    _soc_boundary = bess_properties["soc_boundary_percent"]# percent
    _Pb_boundary  = bess_properties["power_boundary_Erate"]# Erate unit is hour^(-1)
    _Eb_max_puh   = bess_properties["energy_capacity_puh"] # puh 
    self.c_eff    = bess_properties["charging_efficiency"] # energy stored / energy supplied
    self.d_eff    = bess_properties["discharging_efficiency"] # energy output / energy consumed
    self.Eb_max   = hlp.puh_to_pus(_Eb_max_puh) # puh -> pus
    self.initial_soc_space = np.linspace(_soc_boundary.liminf, _soc_boundary.limsup, soc_levels) # percent
    self.Eb_boundary = hlp.calc_Eb_boundary(self.Eb_max, _soc_boundary) # percent -> pus (+, +)
    self.Pb_boundary = hlp.calc_Pb_boundary(_Eb_max_puh, _Pb_boundary) # Erate -> pu (+, -)

    # define observation space
    obs_space_spec = {
        "pv_forecast": (0,1,(1,)),
        "previous_pv_forecast" : (0,1,(1,)),
        "previous_pv_potential": (0,1,(1,)),
        "initial_soc_divided_by_100": (_soc_boundary.liminf / 100,_soc_boundary.limsup / 100,(1,)),
    }
    self.observation_space = spaces.Dict(
        {
          key: spaces.Box(lower_limit, upper_limit, shape=shape, dtype=np.float32) 
          for key, (lower_limit, upper_limit, shape) in obs_space_spec.items()
        }
    )

    # define action space
    self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    # initialize variables needed for evaluation
    if self.render_mode == "evaluation":
      os.makedirs(config.dir_names['evaluation_output'], exist_ok=True)
      self.csv_path = csv_path
      self.max_steps = len(self.nonzero_indices)
      # self.cluster_head_indices = hlp.beginning_of_nonzero_cluster_indices(self.nonzero_indices) # indexing the beginning of each day
      self._cluster_counter = 0 # indexing cluster_head_indices
      self._first_cluster = True
      self._evaluation_flag = True
      
   
  def _get_obs(self):
      # note: pv_potential here means the pv power estimated or measured available to the system,
      # i.e., PV power output in the case of no curtailment
      return {
          "pv_forecast": np.array([self.state["pv_forecast"]], dtype=np.float32), 
          "previous_pv_forecast": np.array([self.previous_state["pv_forecast"]], dtype=np.float32),
          "previous_pv_potential": np.array([self.previous_state["pv_potential"]], dtype=np.float32),
          "initial_soc_divided_by_100": 0.01 * np.array([self.state["initial_soc"]], dtype=np.float32),
      }

  def _get_info(self):
    return {"previous_state": self.previous_state, 
            "current_state": self.state,
            "max_steps": self.max_steps,
            "last_cluster": self.last_cluster,
    }

  def reset(self, seed=None, options=None):
    # We need the following line to seed self.np_random
    super().reset(seed=seed)

    if self._evaluation_flag:
      self.index = self.cluster_head_indices[self._cluster_counter]
      if self._first_cluster:
        initial_soc = config.initial_soc_for_evaluation 
        self._first_cluster = False
      else:
        initial_soc = self.state["initial_soc"]

      if self._cluster_counter < len(self.cluster_head_indices) - 1:
        self._cluster_counter += 1
      else:
        print("env.reset(): Last cluster started.")
        self.last_cluster = True
    else:
      # not evaluating, reset() randomly selects initial states for a new episode
      self.index = self.np_random.choice(self.cluster_head_indices)
      initial_soc = self.np_random.choice(self.initial_soc_space) 

    # note: when reset() is called, the purpose should be 
    # updating self.index to point to the beginning of the next nonzero cluster in PV SCADA sequence
    # which implies self.index-1 points to Pm=0  
    # I assume Pm=0 means the PV-BESS system is idling => no change in SoC
    # and therefore previous_initial_soc = initial_soc.    
    previous_initial_soc = initial_soc
    t, Pf, Pm = self.data[self.index,:]
    previous_Pf, previous_Pm = self.data[self.index-1,1:]
    
    # Initialize current and previous states
    # note: if reset() is called not for the first time, there
    self.state = {
        "datetime": t,
        "initial_soc": initial_soc,  
        "pv_forecast": Pf,
        "pv_potential": Pm,
        "bess_power": None,
        "pv_power": None
    }

    self.previous_state = {
        "datetime": None,
        "initial_soc": previous_initial_soc,  
        "pv_forecast": previous_Pf,
        "pv_potential": previous_Pm,
        "bess_power": None,  
        "pv_power": None    
    }

    # assign for return
    observation = self._get_obs()
    info        = self._get_info()

    return observation, info

  def step(self, action: np.ndarray):
    # execute action
    reward, next_soc, Ppv, Pb, Pnet, actual_c = self._pvbess_dynamics(action)
    
    # update self.state
    self.state["pv_power"]   = Ppv
    self.state["bess_power"] = Pb

    # self.state should be filled with data now.

    # check if there is not more step. True if the next Pm is zero in data record.
    terminated = self.data[self.index+1, -1] == 0

    # update env states for observation and info
    # 1.) store the current state as previous state
    self.previous_state = copy.deepcopy(self.state)

    # 2.) increment the index for reading data
    self.index += 1
    t, Pf, Pm = self.data[self.index,:]
    
    # 3.) fill what is known about the new state. 
    # The next call to step() will update None's of this state dict.
    # If reset() is called the next, self.state["initial_soc"] is kept, Pf and Pm will be different.
    self.state = {
        "datetime": t,
        "initial_soc": next_soc,  
        "pv_forecast": Pf,
        "pv_potential": Pm,
        "bess_power": None,
        "pv_power": None
    }

    # prepare return values
    observation = self._get_obs()
    info        = self._get_info()
    
    # log info and action if evaluating the model
    if self._evaluation_flag:
      self._log_step(info, action, Pnet, reward, actual_c)

    # truncation 
    truncated = True if terminated else False

    # return values as required by Gymnasium API description
    return observation, reward, terminated, truncated, info

  # model considering PV curtailment and prioritize using curtailment
  def _pvbess_dynamics(self, action: np.ndarray):
    Pm   = self.state["pv_potential"]
    Pdfc, c = action # c is the curtailment factor to mean the portion of power planned to be curtailed
    # Pgap = Pdfc - Pm * (1 - c)
    # PV_deficit = Pgap > 0 # whether PV power is sufficient for achieving Pdfc based on the assumption that available PV will be Pm * (1 - c)
    PV_deficit = Pdfc > Pm * (1 - c)
    # calculate Ppv, Pb (postive means charging, negative means discharging)
    if PV_deficit:
      if Pm > Pdfc:
        Ppv = Pdfc # and this implies Pb = 0
      else:
        Ppv = Pm
      Pb, next_soc = self._bess_dynamics(Ppv, Pdfc)
    else:
      # in the case of PV power surplus BESS try to abosrb the surplus power, excess PV power is curtailed
      Pb, next_soc = self._bess_dynamics(Pm, Pdfc)
      Ppv = Pdfc + Pb

    def calculate_reward(Pnet, Pdfc, Pb, c, actual_c):
#     # Pdfc is actor's output
      # mse  = hlp.mse_loss_np(Pnet, Pdfc)
      rmse = hlp.rmse_loss_np(Pnet, Pdfc)  
      net_out_close_to_dfc = np.isclose(Pnet, Pdfc, rtol=0.05)
      net_out_not_close_to_dfc = not net_out_close_to_dfc
      # reward = net_out_close_to_dfc * (Pdfc != 0) - mae - actual_c
      # reward = net_out_close_to_dfc * (Pdfc != 0) - rmse - actual_c - net_out_not_close_to_dfc
      # reward = net_out_close_to_dfc * (Pdfc != 0) - mse - 1.5*actual_c - net_out_not_close_to_dfc
      # reward = net_out_close_to_dfc * (Pdfc != 0) * (actual_c < 0.1) - mse - 1.5*actual_c - net_out_not_close_to_dfc
      # reward = net_out_close_to_dfc * (actual_c < 0.05) - mse - actual_c - net_out_not_close_to_dfc + np.isclose(actual_c, c, rtol=0.05)
      # reward = (1 - rmse) * net_out_close_to_dfc * (1 - actual_c) # reward fn n1
      # reward = (1 - rmse) * net_out_close_to_dfc * (1 - actual_c) - actual_c # reward fn n2
      reward = (1 - rmse) * net_out_close_to_dfc * (1 - actual_c) - actual_c - rmse * net_out_not_close_to_dfc # reward fn n3
      
      return reward

    # calculate reward
    Pnet = Ppv - Pb
    actual_c = 1 - Ppv / Pm
    reward = calculate_reward(Pnet, Pdfc, Pb, c, actual_c)
    return float(reward), next_soc, Ppv, Pb, Pnet, actual_c
  
  def _bess_dynamics(self, Ppv, Pdfc):
    soc = self.state["initial_soc"] # percentage
    Eb = hlp.percent_to_fraction(soc) * self.Eb_max # pus, battery energy

    # impleemnting Eq. (2a, 2b, 3a, 3b, 4) in DeepComp journal article
    # note: charging power P_c and discharging power P_d are positive in DeepComp's BESS model
    
    # DeepComp Eq.(2a)
    P_c_lim = min(self.Pb_boundary.limsup, 
                  (1/self.c_eff) * (self.Eb_boundary.limsup - Eb) / self.sec_per_step)

    # DeepComp Eq.(2b)
    # invert the sign of self.Pb_boundary.liminf (-) to suit the model where P_d is positive
    # (battery power (-) implies discharging for the battery, while P_d (+) means "discharging power")
    P_d_lim = min((-1)*self.Pb_boundary.liminf, 
                  self.d_eff * (Eb - self.Eb_boundary.liminf) / self.sec_per_step)
    
    # DeepComp Eq.(3a)
    P_c = min(max(Ppv - Pdfc, 0), P_c_lim)
    # DeepComp Eq.(3b)
    P_d = min(max(Pdfc - Ppv, 0), P_d_lim)
    # DeepComp Eq.(4)
    next_Eb = Eb + self.c_eff * P_c * self.sec_per_step - (1/self.d_eff) * P_d * self.sec_per_step

    # prepare return values
    Pb = P_c - P_d
    next_soc = hlp.fraction_to_percent(next_Eb / self.Eb_max)
    return Pb, next_soc
  
  def _log_step(self, info, action, Pnet, r, actual_c):
    # Gather the frame of data that is just completed via simulation
    Pdfc, c = action
    log_data = {
        "env_state": info["previous_state"],
        "Pdfc": Pdfc,
        "Pnet": Pnet,
        "cr": c,
        "actual_cr": actual_c,
        "reward": r,
    }

    # flatten dict for csv.DictWriter
    flattened_data = hlp.flatten_nested_dict_to_dict(log_data)
    df = pd.DataFrame([flattened_data])
    # save dict to a list buffer
    file_exist = os.path.isfile(self.csv_path)
    # save to csv file
    df.to_csv(self.csv_path, mode='a', header=not file_exist, index=False)

  def close(self):
    print(f"env.close(): Nothing to be done. Environment should be closed by now.")

  def render(self, mode="human"):
    pass
