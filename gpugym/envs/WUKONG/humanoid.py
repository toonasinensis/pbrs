"""
Environment file for "fixed arm" (FA) humanoid environment
with potential-based rewards implemented
"""

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from gpugym.utils.math import *
from gpugym.envs import LeggedRobot


class Humanoid(LeggedRobot):

    def _custom_init(self, cfg):
        self.dt_step = self.cfg.sim.dt * self.cfg.control.decimation
        self.pbrs_gamma = 0.99
        self.phase = torch.zeros(
            self.num_envs, 1, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.eps = 0.2
        self.phase_freq = 1.

    def compute_observations(self):
        base_z = self.root_states[:, 2].unsqueeze(1)*self.obs_scales.base_z
        in_contact = torch.gt(
            self.contact_forces[:, self.feet_indices, 2], 0).int()
        in_contact = torch.cat(
            (in_contact[:, 0].unsqueeze(1), in_contact[:, 1].unsqueeze(1)),
            dim=1)
        self.commands[:, 0:2] = torch.where(
            torch.norm(self.commands[:, 0:2], dim=-1, keepdim=True) < 0.5,
            0., self.commands[:, 0:2].double()).float()
        self.commands[:, 2:3] = torch.where(
            torch.abs(self.commands[:, 2:3]) < 0.5,
            0., self.commands[:, 2:3].double()).float()
        self.obs_buf = torch.cat((
            base_z,                                 # [1] Base height
            self.base_lin_vel,                      # [3] Base linear velocity
            self.base_ang_vel,                      # [3] Base angular velocity
            self.projected_gravity,                 # [3] Projected gravity
            self.commands[:, 0:3],                  # [3] Velocity commands
            self.smooth_sqr_wave(self.phase),       # [1] Contact schedule
            torch.sin(2*torch.pi*self.phase),       # [1] Phase variable
            torch.cos(2*torch.pi*self.phase),       # [1] Phase variable
            self.dof_pos,                           # [12] Joint states
            self.dof_vel,                           # [12] Joint velocities
            in_contact,                             # [2] Contact states
        ), dim=-1)
        if self.add_noise:
            self.obs_buf += (2*torch.rand_like(self.obs_buf) - 1) \
                * self.noise_scale_vec

    def _get_noise_scale_vec(self, cfg):
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        print("num_actions")
        print(self.num_actions)
        noise_list = [
            torch.ones(1) * noise_scales.base_z * self.obs_scales.base_z,
            torch.ones(3) * noise_scales.lin_vel * self.obs_scales.lin_vel,
            torch.ones(3) * noise_scales.ang_vel * self.obs_scales.ang_vel,
            torch.ones(3) * noise_scales.gravity,
            torch.zeros(3),  # command
            torch.zeros(3),  # command

            torch.ones(self.num_actions) * noise_scales.dof_pos * self.obs_scales.dof_pos,
            torch.ones(self.num_actions) * noise_scales.dof_vel * self.obs_scales.dof_vel,
            torch.zeros(2)
        ]
        leveled_noise_list = [v.to(self.obs_buf.device).to(self.obs_buf.dtype) * noise_level for v in noise_list]
        noise_vec = torch.cat(leveled_noise_list, dim=-1)
        noise_vec = noise_vec.to(self.obs_buf.dtype).to(self.obs_buf.device)
        return noise_vec

        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0] = noise_scales.base_z * self.obs_scales.base_z
        noise_vec[1:4] = noise_scales.lin_vel
        noise_vec[4:7] = noise_scales.ang_vel
        noise_vec[7:10] = noise_scales.gravity
        noise_vec[10:16] = 0.   # commands
        noise_vec[16:26] = noise_scales.dof_pos
        noise_vec[26:36] = noise_scales.dof_vel
        noise_vec[36:38] = noise_scales.in_contact  # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements \
                * noise_level \
                * self.obs_scales.height_measurements
        noise_vec = noise_vec * noise_level
        return noise_vec


    def _custom_reset(self, env_ids):
        if self.cfg.commands.resampling_time == -1:
            self.commands[env_ids, :] = 0.
        self.phase[env_ids, 0] = torch.rand(
            (torch.numel(env_ids),), requires_grad=False, device=self.device)
    
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rb_states_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.root_pos_trajs = self.root_states[:, :3].unsqueeze(-1).expand(-1, -1,
                                                                           int(self.cfg.rewards.mean_vel_window / self.dt)).contiguous()

        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]

        # self.randomize_lag_indices = torch.randint(0, self.cfg.domain_rand.lag_timesteps + 1, (self.num_envs,))
        # self.lag_buffer = torch.zeros(self.num_envs, self.cfg.domain_rand.lag_timesteps + 1, self.num_actions,
        #                               dtype=torch.float, device=self.device, requires_grad=False)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1,
                                                                            3)  # shape: num_envs, num_bodies, xyz axis

        self.foot_contact_forces_prev = self.contact_forces[:, self.feet_indices, :]
        # shape: num_envs, num_bodies, 13 = 7(pos xyz and quat) + 6(lin_vel and ang_vel)
        self.rb_states = gymtorch.wrap_tensor(rb_states_tensor).view(self.num_envs, -1, 13)

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                   requires_grad=False)

        self.ctrl_hist = torch.zeros(self.num_envs, self.num_actions*3,
                                     dtype=torch.float, device=self.device,
                                     requires_grad=False)


        self.torques_prev = self.torques
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.before_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                               requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                    device=self.device, requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.is_halfway_resample = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool, requires_grad=False)
        self.cmd_w = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.exp_dist_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float, requires_grad=False)

        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device, requires_grad=False, )  # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.projected_gravity = self.projected_gravity / torch.sum(self.projected_gravity)

        self.dof_dict = {n: self.dof_names.index(n) for n in self.dof_names}
        if hasattr(self.cfg.normalization, "obs_dof_order"):
            obs_dof_order = self.cfg.normalization.obs_dof_order
            self.obs_dof_indices = [self.dof_dict[name] for name in obs_dof_order]
            self.obs_dof_indices = torch.Tensor(data=self.obs_dof_indices).to(self.device).to(torch.long)
        else:
            self.obs_dof_indices = None

        if hasattr(self.cfg.normalization, "action_dof_order"):
            action_dof_order = self.cfg.normalization.action_dof_order
            self.action_dof_indices = [self.dof_dict[name] for name in action_dof_order]
            self.action_dof_indices = torch.Tensor(data=self.action_dof_indices).to(self.device).to(torch.long)
        else:
            self.action_dof_indices = None

        # joint positions offsets and PD gains
        self.dof_scale = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle

            if self.action_dof_indices is not None:
                for dof_name in self.cfg.control.action_scale.keys():
                    if dof_name in name:
                        self.dof_scale[i] = self.cfg.control.action_scale[dof_name]

            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.dof_scale = self.dof_scale.unsqueeze(0)

        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.body_vel_buf =  torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        #read dof arrage
        if self.cfg.init_state.reset_mode == "reset_to_range":
            self.dof_pos_range = torch.zeros(self.num_dof, 2,
                                            dtype=torch.float,
                                            device=self.device,
                                            requires_grad=False)
            self.dof_vel_range = torch.zeros(self.num_dof, 2,
                                            dtype=torch.float,
                                            device=self.device,
                                            requires_grad=False)

            for joint, vals in self.cfg.init_state.dof_pos_range.items():
                for i in range(self.num_dof):
                    if joint in self.dof_names[i]:
                        self.dof_pos_range[i, :] = to_torch(vals)

            for joint, vals in self.cfg.init_state.dof_vel_range.items():
                for i in range(self.num_dof):
                    if joint in self.dof_names[i]:
                        self.dof_vel_range[i, :] = to_torch(vals)

            self.root_pos_range = torch.tensor(self.cfg.init_state.root_pos_range,
                    dtype=torch.float, device=self.device, requires_grad=False)
            self.root_vel_range = torch.tensor(self.cfg.init_state.root_vel_range,
                    dtype=torch.float, device=self.device, requires_grad=False)
        # self.obs_history_buf = torch.zeros(self.num_envs, self.history_buffer_len, self.num_obs, device=self.device,
        #                                    dtype=torch.float)
        # self.obs_history = torch.zeros(self.num_envs, self.num_obs_history, device=self.device, dtype=torch.float)

        # if self.cfg.terrain.measure_heights:
        #     self.height_points = self._init_height_points()
        #     self.base_height_points = self._init_base_height_points()
        #     self.foot_height_points = self._init_foot_height_points()
        self.measured_heights = 0
        self.measured_foot_heights = 0
        # self.base_height_buf = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float)
        # self.foot_heightmap_buf = torch.zeros(self.num_envs, self.num_foot_height_points, device=self.device,
                                            #   dtype=torch.float)



    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        self.joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                            requires_grad=False)

        # if self.cfg.domain_rand.randomize_lag_timesteps:
        #     self.lag_buffer = torch.cat((self.lag_buffer[:, 1:], actions.clone().unsqueeze(1)), dim=1)
        #     # self.randomize_lag_indices = torch.clip(self.randomize_lag_indices + torch.randint(-1, 2, (self.num_envs,)), 0, self.cfg.domain_rand.lag_timesteps)
        #     this_action = self.lag_buffer[torch.arange(self.num_envs), self.randomize_lag_indices]
        # else:
        this_action = actions

        if self.action_dof_indices is not None:
            self.joint_pos_target[:, self.action_dof_indices] = this_action * self.dof_scale[:, self.action_dof_indices]
        else:
            self.joint_pos_target[:, :self.num_actions] = this_action * self.cfg.control.action_scale

        self.joint_pos_target = self.joint_pos_target + self.default_dof_pos

        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = (self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) -
                       self.d_gains * self.Kd_factors * self.dof_vel)
        elif control_type == "V":
            torques = (self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_vel) -
                       self.d_gains * self.Kd_factors * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt)
        elif control_type == "T":
            torques = self.joint_pos_target
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        # torques = torques * self.motor_strengths

        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    def _post_physics_step_callback(self):
        self.phase = torch.fmod(self.phase + self.dt, 1.0)
        env_ids = (
            self.episode_length_buf
            % int(self.cfg.commands.resampling_time / self.dt) == 0) \
            .nonzero(as_tuple=False).flatten()
        if self.cfg.commands.resampling_time == -1 :
            # print(self.commands)
            pass  # when the joystick is used, the self.commands variables are overridden
        else:
            self._resample_commands(env_ids)
            if (self.cfg.domain_rand.push_robots and
                (self.common_step_counter
                % self.cfg.domain_rand.push_interval == 0)):
                self._push_robots()

    def _push_robots(self):
        # Randomly pushes the robots.
        # Emulates an impulse by setting a randomized base velocity.
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:8] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 1), device=self.device)
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def check_termination(self):
        """ Check if environments need to be reset
        """
        # Termination for contact
        term_contact = torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :],
                dim=-1)
        self.reset_buf = torch.any((term_contact > 1.), dim=1)

        # Termination for velocities, orientation, and low height
        self.reset_buf |= torch.any(
          torch.norm(self.base_lin_vel, dim=-1, keepdim=True) > 10., dim=1)
        self.reset_buf |= torch.any(
          torch.norm(self.base_ang_vel, dim=-1, keepdim=True) > 5., dim=1)
        self.reset_buf |= torch.any(
          torch.abs(self.projected_gravity[:, 0:1]) > 0.7, dim=1)
        self.reset_buf |= torch.any(
          torch.abs(self.projected_gravity[:, 1:2]) > 0.7, dim=1)
        self.reset_buf |= torch.any(self.base_pos[:, 2:3] < 0.3, dim=1)

        # # no terminal reward for time-outs
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

# ########################## REWARDS ######################## #

    # * "True" rewards * #

    def _reward_tracking_lin_vel(self):
        # Reward tracking specified linear velocity command
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        error *= 1./(1. + torch.abs(self.commands[:, :2]))
        error = torch.sum(torch.square(error), dim=1)
        return torch.exp(-error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Reward tracking yaw angular velocity command
        ang_vel_error = torch.square(
            (self.commands[:, 2] - self.base_ang_vel[:, 2])*2/torch.pi)
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    # * Shaping rewards * #

    def _reward_base_height(self):
        # Reward tracking specified base height
        base_height = self.root_states[:, 2].unsqueeze(1)
        error = (base_height-self.cfg.rewards.base_height_target)
        error *= self.obs_scales.base_z
        error = error.flatten()
        return torch.exp(-torch.square(error)/self.cfg.rewards.tracking_sigma)

    def _reward_orientation(self):
        # Reward tracking upright orientation
        error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return torch.exp(-error/self.cfg.rewards.tracking_sigma)

    def _reward_dof_vel(self):
        # Reward zero dof velocities
        dof_vel_scaled = self.dof_vel/self.cfg.normalization.obs_scales.dof_vel
        return torch.sum(self.sqrdexp(dof_vel_scaled), dim=-1)

    def _reward_joint_regularization(self):
        # Reward joint poses and symmetry
        error = 0.
        # Yaw joints regularization around 0
        error += self.sqrdexp(
            (self.dof_pos[:, 0]) / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            (self.dof_pos[:, 5]) / self.cfg.normalization.obs_scales.dof_pos)
        # Ab/ad joint symmetry
        error += self.sqrdexp(
            (self.dof_pos[:, 1] - self.dof_pos[:, 6])
            / self.cfg.normalization.obs_scales.dof_pos)
        # Pitch joint symmetry
        error += self.sqrdexp(
            (self.dof_pos[:, 2] + self.dof_pos[:, 7])
            / self.cfg.normalization.obs_scales.dof_pos)
        return error/4

    def _reward_ankle_regularization(self):
        # Ankle joint regularization around 0
        error = 0
        error += self.sqrdexp(
            (self.dof_pos[:, 4]) / self.cfg.normalization.obs_scales.dof_pos)
        error += self.sqrdexp(
            (self.dof_pos[:, 9]) / self.cfg.normalization.obs_scales.dof_pos)
        return error

    # * Potential-based rewards * #

    def pre_physics_step(self):
        self.rwd_oriPrev = self._reward_orientation()
        self.rwd_baseHeightPrev = self._reward_base_height()
        self.rwd_jointRegPrev = self._reward_joint_regularization()

    def _reward_ori_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_orientation() - self.rwd_oriPrev)
        return delta_phi / self.dt_step

    def _reward_jointReg_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_joint_regularization() - self.rwd_jointRegPrev)
        return delta_phi / self.dt_step

    def _reward_baseHeight_pb(self):
        delta_phi = ~self.reset_buf \
            * (self._reward_base_height() - self.rwd_baseHeightPrev)
        return delta_phi / self.dt_step

# ##################### HELPER FUNCTIONS ################################## #

    def sqrdexp(self, x):
        """ shorthand helper for squared exponential
        """
        return torch.exp(-torch.square(x)/self.cfg.rewards.tracking_sigma)

    def smooth_sqr_wave(self, phase):
        p = 2.*torch.pi*phase * self.phase_freq
        return torch.sin(p) / \
            (2*torch.sqrt(torch.sin(p)**2. + self.eps**2.)) + 1./2.
