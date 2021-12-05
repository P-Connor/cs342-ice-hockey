
from image_agent.utils import _to_image
from torchvision.transforms import functional as F
from image_agent import utils
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import copy
import torch
import matplotlib
matplotlib.use('Agg')


DEBUG = True


class Player:
    def __init__(self):
        # Current kart info
        self.location = np.array([0, 0, 0])
        self.velocity = np.array([0, 0, 0])
        self.speed = 0
        self.heading = np.array([0, 0, 0])

        # Persistent player info
        self.stuck_counter = 0
        self.reverse_cooldown = 0

        # Target information
        self.target = {"location": [0, 0, 0], "heading": [0, 0, 1], "speed": 0}
        self.to_target = np.array([0, 0, 0])
        self.angle_to_target = 0
        self.distance_to_target = 0

    def update_state(self, kart_state):
        self.location = np.array(kart_state['location'])
        self.velocity = np.array(kart_state['velocity'])
        self.speed = np.linalg.norm(self.velocity)
        self.heading = kart_state['front'] - self.location
        self.heading /= np.linalg.norm(self.heading)  # normalize

        # TODO tune speed value
        if(self.speed < 0.3):
            self.stuck_counter += 1

        self.update_target(self.target)

    def update_target(self, target):
        self.target = target
        self.to_target = target['location'] - self.location
        self.distance_to_target = np.linalg.norm(self.to_target)
        self.to_target /= np.linalg.norm(self.to_target)  # normalize
        self.angle_to_target = ((np.arccos(np.clip(np.dot(self.heading, self.to_target), -1.0, 1.0)) / math.pi)
                                * np.sign(np.cross(self.heading, self.to_target)[1]))

    # Computes the drive actions to reach the player's current target based on its current state
    # and returns the actions as a dict

    def drive(self):
        # Handle steering/drifting
        steer = self.angle_to_target * 8
        drift = False

        # Calculate acceleration/braking
        brake = abs(self.angle_to_target) > 0.2
        acceleration = 0
        if(brake):
            self.reverse_cooldown = 4
            steer = -steer
        elif self.reverse_cooldown > 0:
            steer = -steer
            self.reverse_cooldown -= 1
        else:
            acceleration = (1 - abs(self.angle_to_target) * 5) + 0.1

        # Reset if stuck (handle this better later)
        # rescue = acceleration > 0.3 and np.linalg.norm(player['kart']['velocity']) < 0.05
        rescue = False
        nitro = False
        fire = False

        return {
            "acceleration": acceleration,
            "brake": brake,
            "drift": drift,
            "fire": fire,
            "nitro": nitro,
            "rescue": rescue,
            "steer": steer
        }

    # Returns a score based on how easily this player can reach its current target.
    # The lower the score the quicker the player will be able to reach.
    # If this is below a certain threshold the target can be considered already reached.
    def score(self):
        # TODO calculate better score
        return self.distance_to_target

    # Returns a score based on how easily this player can reach the specified potential target.
    # The lower the score the quicker the player will be able to reach.
    # If this is below a certain threshold the target can be considered already reached.
    def try_target(self, target):
        old_target = copy.deepcopy(self.target)
        self.target = target
        score = self.score()
        self.target = old_target
        return score

    # Plots this current player and their target info to the primary matplotlib plot
    def plot(self):
        plt.plot(self.location[0], self.location[2], ".")
        plt.plot([self.location[0], self.location[0] + self.heading[0] * 3],
                 [self.location[2], self.location[2] + self.heading[2] * 3])
        plt.plot(self.target['location'][0], self.target['location'][2], "x")
        plt.plot([self.target['location'][0], self.target['location'][0] + self.target['heading'][0] * 3],
                 [self.target['location'][2], self.target['location'][2] + self.target['heading'][2] * 3])
        plt.xticks(range(-50, 60, 10))
        plt.yticks(range(-60, 70, 10))


class Team:
    agent_type = 'image'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None
        self.model = utils.load_model()
        self.players = []
        if DEBUG:
            self.frame = 0
            plt.figure(figsize=(12, 6), dpi=80)

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
        self.players = [Player() for i in range(num_players)]
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
                          for img in player_image])

        pos = torch.cat([torch.tensor(player_state[i]['kart']['location']).unsqueeze(
            0) for i in range(len(player_state))])

        pos_puck_location = self.model(imgs, pos)

        # checks results from model. If puck position unknown, set to None
        puck_location = None
        if pos_puck_location[0][1] <= -0.5:
            puck_location = pos_puck_location[0]
        elif pos_puck_location[1][1] <= -0.5:
            puck_location = pos_puck_location[1]

        if puck_location is not None:
            puck_location = puck_location.detach().numpy()

        import matplotlib.pyplot as plt
        import torchvision.transforms.functional as TF
        img = player_image[0]
        print(img.shape)
        pred = _to_image(puck_location.detach().numpy(), player_state[0]['camera']
                         ['projection'], player_state[0]['camera']['view'])
        fig, ax = plt.subplots(1, 1)
        ax.imshow(TF.to_pil_image(img))

        WH2 = np.array([img.shape[1], img.shape[0]])/2
        ax.add_artist(plt.Circle(
            WH2*(pred+1), 2, ec='r', fill=False, lw=1.5))

        ret = []
        assert(len(self.players) == len(player_state))
        for i, (player, state) in enumerate(zip(self.players, player_state)):
            # self.model(player_image, player_state[0]['kart']['location'])

            # TODO: Calculate a drive target based on puck location and other variables
            # target = self.test_target

            player.update_state(state['kart'])

            if(player.score() < 3):
                player.update_target({
                    "location": [random.randint(-20, 20), 0, random.randint(-20, 20)],
                    "heading": [random.randint(-1, 1), 0, random.randint(-1, 1)],
                    "speed": [random.randint(0, 10)]
                })

            if DEBUG:
                plt.subplot(1, len(self.players), i + 1)
                player.plot()
                if(puck_location is not None):
                    plt.plot(puck_location[0], puck_location[2], "h")
                    plt.xticks(range(-50, 60, 10))
                    plt.yticks(range(-60, 70, 10))

            ret.append(player.drive())

        if DEBUG:
            plt.savefig("plot/" + str(self.frame) + ".png")
            self.frame += 1
            plt.clf()

        # ret = [{"acceleration": .1, "steer": 0 if len(self.x) < 20 else 1}, {"acceleration": 0, "steer": 1}]
        # self.x.append(player_state[0]['kart']['location'][0])
        # self.z.append(player_state[0]['kart']['location'][2])
        # plt.plot(self.x, self.z)
        # plt.savefig("temp.png")

        return ret
