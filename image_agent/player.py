
import numpy as np
import math
import random
import torch
from grader import utils
from torchvision.transforms import functional as F


class Team:
    agent_type = 'image'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None
        self.test_target = [20, 0, 20]
        self.model = utils.load_model()

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.team, self.num_players = team, num_players
        return ['tux'] * num_players

    def act(self, player_state, player_image):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             # pystk.Player>.
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html
                             See HW5 for some inspiration on how to use the camera information.
                             camera:  Camera info for each player
                               - aspect:     Aspect ratio
                               - fov:        Field of view of the camera
                               - mode:       Most likely NORMAL (0)
                               - projection: float 4x4 projection matrix
                               - view:       float 4x4 view matrix
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param player_image: list[np.array] showing the rendered image from the viewpoint of each kart. Use
                             player_state[i]['camera']['view'] and player_state[i]['camera']['projection'] to find out
                             from where the image was taken.

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """

        imgs = torch.cat([F.to_tensor(img).unsqueeze(0)
                          for img in player_image])  # .unsqueeze(0)

        pos = torch.cat([torch.tensor(player_state[i]['kart']['location']).unsqueeze(
            0) for i in range(len(player_state))])

        pos_puck_location = self.model(imgs, pos)

        # chesks results from model. If puck position unknown, set to None
        puck_location = None
        if pos_puck_location[0][1] != -1:
            puck_location = pos_puck_location[0]
        elif pos_puck_location[1][1] != -1:
            puck_location = pos_puck_location[1]

        ret = []
        for player in player_state:
            # self.model(player_image, player_state[0]['kart']['location'])
            # Populate this with the actions we will take for this player
            actions = dict()

            # TODO: Calculate a drive target based on puck location and other variables
            target = self.test_target

            # Get some preliminary information
            current_location = np.array(player['kart']['location'])
            current_location[1] = 0   # Remove y coordinate
            distance_to_target = np.linalg.norm(target - current_location)

            # Find angle_to_target, mapped from -1 to 1
            facing = player['kart']['front'] - current_location
            facing[1] = 0   # Remove y coordinate
            facing_u = facing / np.linalg.norm(facing)
            to_target = target - current_location
            to_target_u = to_target / np.linalg.norm(to_target)
            angle_to_target = (np.arccos(np.clip(np.dot(facing_u, to_target_u), -1.0, 1.0)
                                         ) / math.pi) * np.sign(np.cross(facing_u, to_target_u)[1])
            # print(angle_to_target, distance_to_target)

            # Handle steering/drifting
            actions['steer'] = angle_to_target * 5
            actions['drift'] = 0.2 < abs(angle_to_target) < 0.4

            # Calculate acceleration/braking
            actions['acceleration'] = 1 - abs(angle_to_target) * 3
            actions['brake'] = abs(angle_to_target) > 0.5

            # Reset if stuck (handle this better later)
            actions['rescue'] = actions['acceleration'] > 0.3 and np.linalg.norm(
                player['kart']['velocity']) < 0.05

            if(distance_to_target < 3):
                self.test_target = [
                    random.randint(-20, 20), 0, random.randint(-20, 20)]

            ret.append(actions)

        return ret
