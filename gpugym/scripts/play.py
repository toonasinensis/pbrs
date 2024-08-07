# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from gpugym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from gpugym.envs import *
from gpugym.utils import  get_args, export_policy, export_critic, task_registry, Logger

import numpy as np
import torch
import numpy as np
import torch
from torch.utils import data


def play(args):
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.commands.ranges.lin_vel_x = [1,1]
    env_cfg.commands.ranges.lin_vel_y = [0,0]
    env_cfg.commands.ranges.ang_vel_yaw = [0,0]
    env_cfg.commands.ranges.heading = [0,0]
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False #True
    env_cfg.domain_rand.push_interval_s = 2
    env_cfg.domain_rand.max_push_vel_xy = 1.0
    env_cfg.init_state.reset_ratio = 0.8

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # TRAINING COMMAND RANGES #

            # lin_vel_x = [0, 2.5]        # min max [m/s]
            # lin_vel_y = [-0.75, 0.75]   # min max [m/s]
            # ang_vel_yaw = [-2., 2.]     # min max [rad/s]
            # heading = [0., 0.]


    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy(ppo_runner.alg.actor_critic, path)
        print('Exported policy model to: ', path)

    # export critic as a jit module (used to run it from C++)
    if EXPORT_CRITIC:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'critics')
        export_critic(ppo_runner.alg.actor_critic, path)
        print('Exported critic model to: ', path)

    logger = Logger(env.dt)
    robot_index = 4  # which robot is used for logging
    joint_index = 2  # which joint is used for logging
    stop_state_log = 1000  # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    print(env.commands)
    play_log = []
    dict_list = []
    dof_pos_key = []
    dof_tau_key = []
    dof_vel_key = []
    for j in range(12):
        dof_pos_key.append('dof_pos'+'_'+str(j+1))
        dof_vel_key.append('dof_vel'+'_'+str(j+1))
        dof_tau_key.append('dof_torque'+'_'+str(j+1))

    gravatiy_key = ['gravity_x','gravity_y','gravity_z']
    base_v_key = ['vx','vy','vz']
    base_w_key = ['wx','wy','wz']
    contact_F_key = ['contact_forces_z_L','contact_forces_z_R']
    env.max_episode_length = 100./env.dt
    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 1:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)
        

        # print(dof_tau_key)
        if i < stop_state_log:
            dic_to_save = {}
            # dof_key = ['dof_pos','dof_vel','dof_torque']
            
            
            # for i in range(len(dof_key)):
            for j in range(12):
                
                dic_to_save['dof_pos'+'_'+str(j+1)] = env.dof_pos[robot_index, j].item()
                dic_to_save['dof_vel'+'_'+str(j+1)] = env.dof_vel[robot_index, j].item()
                dic_to_save['dof_torque'+'_'+str(j+1)] = env.torques[robot_index, j].item()

            for i in range(len(gravatiy_key)):
                dic_to_save[gravatiy_key[i]] = env.projected_gravity[robot_index,i].item()

            for i in range(len(base_v_key)):
                dic_to_save[base_v_key[i]] = env.base_lin_vel[robot_index,i].item()
                dic_to_save[base_w_key[i]] = env.base_ang_vel[robot_index,i].item()
            
            
            dic_to_save['height_z'] =   env.root_states[robot_index, 2].item()

            # dic_to_save['contact_forces_z'] = env.base_ang_vel[robot_index,2].item()

            # dic_to_save[]
            # print(env.contact_forces[robot_index, :, 2])
            dic_to_save['contact_forces_z_L'] =env.contact_forces[robot_index, env.feet_indices[0], 2].cpu().item()
            dic_to_save['contact_forces_z_R'] =env.contact_forces[robot_index, env.feet_indices[1], 2].cpu().item()
            # print(env.contact_forces[robot_index, env.feet_indices[1], 2].cpu().item())
            logger.log_states(
                {

                    'actions': actions[robot_index, joint_index].item(),
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    # 'gravity_xyz': env.projected_gravity[robot_index,:].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'height_z': env.root_states[robot_index, 2].item(),
                }
            )
            dict_list.append(dic_to_save)

            ### Humanoid PBRS Logging ###
            # [ 1]  Timestep
            # [38]  Agent observations
            # [10]  Agent actions (joint setpoints)
            # [13]  Floating base states in world frame
            # [ 6]  Contact forces for feet
            # [10]  Joint torques
            play_log.append(
                [i*env.dt]
                + obs[robot_index, :].cpu().numpy().tolist()
                + actions[robot_index, :].detach().cpu().numpy().tolist()
                + env.root_states[0, :].detach().cpu().numpy().tolist()
                + env.contact_forces[robot_index, env.end_eff_ids[0], :].detach().cpu().numpy().tolist()
                + env.contact_forces[robot_index, env.end_eff_ids[1], :].detach().cpu().numpy().tolist()
                + env.torques[robot_index, :].detach().cpu().numpy().tolist()
            )
        elif i==stop_state_log:
            filepath = '/home/xie/Desktop/pbrs-humanoid/experiment/play_log.csv'
            logger.write_states(dict_list,filepath)
            dict_list= logger.read_dicts_from_csv(filepath)
            for key,value in dict_list[0].items():
                print(key,value)
            print("dict_list", len(dict_list))
            data_all = len(dict_list)-1
            obs_num = len(dof_pos_key)+len(dof_tau_key)+len(dof_tau_key)+len(gravatiy_key)\
            +len(base_w_key)+len(contact_F_key)
            est_num = 3+1#vx,vy,vz,hz
            obs_key = contact_F_key+base_w_key+gravatiy_key+dof_tau_key+dof_tau_key+dof_pos_key
            est_key = base_v_key+['height_z']
            feature = np.zeros((data_all,obs_num))
            label = np.zeros((data_all,est_num))
            for i in range(data_all):
                for j in range(len(obs_key)):
                    feature[i,j] = dict_list[i][obs_key[j]]
                for k in range(len(est_key)):
                    label[i,k] = dict_list[i][est_key[k]]
            print("feature",feature[0,:])
            print("label",label[0,:])

            features = torch.tensor([[1, 2], [3, 4], [5, 6]])
            labels = torch.tensor([0, 1, 0])

            # 将数据打包到一个元组中
            data_arrays = (features, labels)

            # 使用 * 符号解包元组，并传递给 TensorDataset
            dataset = data.TensorDataset(*data_arrays)

            # 查看数据集的第一个元素
            print(dataset[0])



            

            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                # if num_episodes>0:
                    # logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    EXPORT_CRITIC = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
