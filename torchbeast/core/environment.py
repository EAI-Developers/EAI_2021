# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The environment class for MonoBeast."""

import torch
import cv2 as cv
from torchbeast.core.pre_attention import get_pre_map
import numpy as np

def _format_frame(frame):
    frame = np.transpose(cv.resize(np.transpose(frame, (1,2,0)), (160, 210)), (2,0,1))
    # print(frame.shape)
    frame = torch.from_numpy(frame)
    return frame.view((1, 1) + frame.shape)  # (...) -> (T,B,...).


class Environment:
    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        # This supports only single-tensor actions ATM.
        initial_last_action = torch.zeros(1, 1, dtype=torch.int64)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8)
        initial_frame = _format_frame(self.gym_env.reset())
        #initial_frame = torch.ByteTensor(cv.resize(initial_frame[0][0].permute(1,2,0).numpy(), (160, 210), interpolation = cv.INTER_AREA)).permute(2, 0, 1).reshape(1,1, -1)
        Res = get_pre_map(initial_frame)
        return dict(
            frame=initial_frame,
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=initial_last_action,
            Enemies = torch.FloatTensor(Res['Enemies']),
            Me = torch.FloatTensor(Res['Me']),
        )

    def step(self, action):
        frame, reward, done, unused_info = self.gym_env.step(action.item())
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if done:
            frame = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        frame = _format_frame(frame)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)
        Res = get_pre_map(frame)

        return dict(
            frame=frame,
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action,
            Enemies = torch.FloatTensor(Res['Enemies']),
            Me = torch.FloatTensor(Res['Me']),
        )

    def close(self):
        self.gym_env.close()
