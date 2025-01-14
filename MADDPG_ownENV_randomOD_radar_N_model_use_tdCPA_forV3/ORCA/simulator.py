import RVOmath as rvo_math
from kdtree import KdTree
from agent import Agent
from obstacle import Obstacle
from vector import Vector2
import math


class Simulator:
    """
    Defines the simulation.
    """

    def __init__(self, globalTime, timestep):
        """
        Constructs and initializes a simulation.
        """
        self.agents_ = []
        self.default_agent_ = None
        self.kd_tree_ = KdTree(self)
        self.obstacles_ = []
        self.global_time_ = globalTime
        self.time_step_ = timestep

    def add_agent(self, position, goalVectors, intruRad, initialVel, maxSpd, attribute=None):
        """
        Adds a new agent with default properties to the simulation.

        Args:
            position (Vector2): The two-dimensional starting position of this agent.

        Returns:
            int: The number of the agent, or -1 when the agent defaults have not been set.
        """
        # if self.default_agent_ is None:
        #     raise Exception('Default agent not set. Call set_agent_defaults first.')

        agent = Agent(self)
        agent.id_ = len(self.agents_)
        agent.max_neighbors_ = 30
        agent.max_speed_ = maxSpd
        agent.neighbor_dist_ = 30  # this is the detection range of a drone
        agent.position_ = position
        agent.radius_ = intruRad
        agent.time_horizon_ = 3
        agent.time_horizon_obst_ = 2  # equals to simulator time step
        agent.velocity_ = initialVel
        agent.goalVectors = goalVectors
        agent.attribute = attribute
        self.agents_.append(agent)

        return agent.id_

    def add_obstacle(self, vertices):
        """
        Adds a new obstacle to the simulation.

        Args:
            vertices (list): List of the vertices of the polygonal obstacle in counterclockwise order.

        Returns:
            int: The number of the first vertex of the obstacle, or -1 when the number of vertices is less than two.

        Remarks:
            To add a "negative" obstacle, e.g. a bounding polygon around the environment, the vertices should be listed in clockwise order.
        """
        if len(vertices) < 2:
            raise Exception('Must have at least 2 vertices.')

        obstacleNo = len(self.obstacles_)

        for i in range(len(vertices)):
            obstacle = Obstacle()
            obstacle.point_ = vertices[i]

            if i != 0:
                obstacle.previous_ = self.obstacles_[len(self.obstacles_) - 1]
                obstacle.previous_.next_ = obstacle

            if i == len(vertices) - 1:
                obstacle.next_ = self.obstacles_[obstacleNo]
                obstacle.next_.previous_ = obstacle

            obstacle.direction_ = rvo_math.normalize(
                vertices[0 if i == len(vertices) - 1 else i + 1] - vertices[i])

            if len(vertices) == 2:
                obstacle.convex_ = True
            else:
                obstacle.convex_ = rvo_math.left_of(
                    vertices[len(vertices) - 1 if i == 0 else i - 1],
                    vertices[i],
                    vertices[0 if i == len(vertices) - 1 else i + 1]) >= 0.0

            obstacle.id_ = len(self.obstacles_)
            self.obstacles_.append(obstacle)

        return obstacleNo

    def step(self):
        """
        Performs a simulation step and updates the two-dimensional position and two-dimensional velocity of each agent.

        Returns:
            float: The global time after the simulation step.
        """

        self.kd_tree_.build_agent_tree()

        # TODO: Try async/await here.
        # Performs a simulation step.
        for agentNo in range(self.num_agents):
            self.agents_[agentNo].compute_neighbors()
            self.agents_[agentNo].compute_new_velocity()

        # TODO: Try async/await here.
        # Updates the two-dimensional position and two-dimensional velocity of each agent.
        for agentNo in range(self.num_agents):
            self.agents_[agentNo].update()

        self.global_time_ += self.time_step_

        return self.global_time_

    @property
    def global_time(self):
        """
        Returns the global time of the simulation.
        """
        return self.global_time_

    @property
    def num_agents(self):
        """
        Returns the count of agents in the simulation.
        """
        return len(self.agents_)

    @property
    def num_obstacles(self):
        """
        Returns the count of obstacles in the simulation.
        """
        return len(self.obstacles_)

    def process_obstacles(self):
        """
        Processes the obstacles that have been added so that they are accounted for in the simulation.

        Remarks:
            Obstacles added to the simulation after this function has been called are not accounted for in the simulation.
        """
        self.kd_tree_.build_obstacle_tree()

    def set_agent_defaults(self, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed, velocity):
        """
        Sets the default properties for any new agent that is added.

        Args:
            neighborDist (float): The default maximum distance (center point to center point) to other agents a new agent takes into account in the navigation. The larger this number, the longer he running time of the simulation. If the number is too low, the simulation will not be safe. Must be non-negative.
            maxNeighbors (int): The default maximum number of other agents a new agent takes into account in the navigation. The larger this number, the longer the running time of the simulation. If the number is too low, the simulation will not be safe.
            timeHorizon (float): The default minimal amount of time for which a new agent's velocities that are computed by the simulation are safe with respect to other agents. The larger this number, the sooner an agent will respond to the presence of other agents, but the less freedom the agent has in choosing its velocities. Must be positive.
            timeHorizonObst (float): The default minimal amount of time for which a new agent's velocities that are computed by the simulation are safe with respect to obstacles. The larger this number, the sooner an agent will respond to the presence of obstacles, but the less freedom the agent has in choosing its velocities. Must be positive.
            radius (float): The default radius of a new agent. Must be non-negative.
            maxSpeed (float): The default maximum speed of a new agent. Must be non-negative.
            velocity (Vector2): The default initial two-dimensional linear velocity of a new agent.
        """
        if self.default_agent_ is None:
            self.default_agent_ = Agent(self)

        self.default_agent_.max_neighbors_ = maxNeighbors
        self.default_agent_.max_speed_ = maxSpeed
        self.default_agent_.neighbor_dist_ = neighborDist
        self.default_agent_.radius_ = radius
        self.default_agent_.time_horizon_ = timeHorizon
        self.default_agent_.time_horizon_obst_ = timeHorizonObst
        self.default_agent_.velocity_ = velocity

    def set_agent_pref_velocity(self, agentNo, prefVelocity):
        """
        Sets the two-dimensional preferred velocity of a specified agent.

        Args:
            agentNo (int): The number of the agent whose two-dimensional preferred velocity is to be modified.
            prefVelocity (Vector2): The replacement of the two-dimensional preferred velocity.
        """
        self.agents_[agentNo].pref_velocity_ = prefVelocity

    def set_pref_vel_newVer(self, agentNo, gridLength):
        oldGoalVector = self.agents_[agentNo].goalVectors
        if oldGoalVector == None:
            print('check')
        if len(oldGoalVector) == 1:
            #self.agents_[agentNo].pref_velocity_ = Vector2(0, 0)
            self.agents_[agentNo].pref_velocity_ = self.agents_[agentNo].velocity_
            return
        newGoalVec = []
        curHeading = math.atan2((oldGoalVector[1][1] - oldGoalVector[0][1]), (oldGoalVector[1][0] - oldGoalVector[0][0]))
        newGoalVec.append(oldGoalVector[0])  # add the starting node
        for id_ in range(2, len(oldGoalVector)):
            nextHeading = math.atan2((oldGoalVector[id_][1] - oldGoalVector[id_ - 1][1]), (oldGoalVector[id_][0] - oldGoalVector[id_ - 1][0]))
            if curHeading != nextHeading:  # add the "id_-1" th element
                newGoalVec.append(oldGoalVector[id_ - 1])
                curHeading = nextHeading  # update the current heading
        newGoalVec.append(oldGoalVector[-1])  # always add the last node
        # update the goal Vector for each agent
        self.agents_[agentNo].goalVectors = newGoalVec
        # set preferred velocity, here always choose "1" for goalVectors.
        prefer_heading = math.atan2((self.agents_[agentNo].goalVectors[1][1] - self.agents_[agentNo].position_.y),
                                    (self.agents_[agentNo].goalVectors[1][0] - self.agents_[agentNo].position_.x))
        self.agents_[agentNo].pref_velocity_ = Vector2(self.agents_[agentNo].max_speed_ * math.cos(prefer_heading),
                                                       self.agents_[agentNo].max_speed_ * math.sin(prefer_heading))

    def set_time_step(self, timeStep):
        """
        Sets the time step of the simulation.

        Args:
            timeStep (float): The time step of the simulation. Must be positive.
        """
        self.time_step_ = timeStep
