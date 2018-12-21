import Box2D as b2
import pygame
import numpy as np

# Extend polygon shape with drawing function
def draw_polygon(polygon, body, fixture, screen, params):
  vertices = [(body.transform * v) * params.PPM for v in polygon.vertices]
  vertices = [(v[0], params.DISPLAY_SIZE[1] - v[1]) for v in vertices]
  pygame.draw.polygon(screen, params.colors[body.type], vertices)

b2.b2.polygonShape.draw = draw_polygon

# Extend circle shape with drawing function
def my_draw_circle(circle, body, fixture, screen, params):
  position = body.transform * circle.pos * params.PPM
  position = (position[0], params.DISPLAY_SIZE[1] - position[1])
  pygame.draw.circle(screen,
                     params.colors[body.type],
                     [int(x) for x in position],
                     int(circle.radius * params.PPM))

b2.b2.circleShape.draw = my_draw_circle

# Params class
class Params(object):
  # Define simulation parameters (Might move them to a param file)
  # The world is centered at the lower left corner of the table
  TABLE_SIZE = np.array([3., 3.])
  TABLE_CENTER = np.array(TABLE_SIZE / 2)
  DISPLAY_SIZE = (600, 600)
  TO_PIXEL = np.array(DISPLAY_SIZE) / TABLE_SIZE

  LINK_0_LENGTH = 1.
  LINK_1_LENGTH = 1.
  LINK_ELASTICITY = 0.
  LINK_FRICTION = .9
  LINK_THICKNESS = 0.05

  BALL_RADIUS = .1
  BALL_ELASTICITY = .5
  BALL_FRICTION = .9

  WALL_THICKNESS = .05
  WALL_ELASTICITY = .95
  WALL_FRICTION = .9

  # Graphic params
  PPM = int(min(DISPLAY_SIZE)/max(TABLE_SIZE))
  TARGET_FPS = 20
  TIME_STEP = 1.0 / TARGET_FPS

  colors = {
    b2.b2.staticBody: (111, 111, 111, 255),
    b2.b2.dynamicBody: (0, 0, 0, 255),
  }


class PhysicsSim(object):

  def __init__(self, balls_pose=[[0, 0]], arm_position=None, params=None):
    self.render = True ## TODO REMOVE
    if params is None:
      self.params = Params()
    else:
      self.params = params

    # Create physic simulator
    self.world = b2.b2World(gravity=(0, 0), doSleep=True)
    self.dt = 1./60
    self.vel_iter = 10
    self.pos_iter = 10
    self._create_table()
    self._create_balls(balls_pose)
    self._create_robotarm(arm_position)
    self._create_holes()

    ## TODO REMOVE
    if self.render:
      self.screen = pygame.display.set_mode((self.params.DISPLAY_SIZE[0], self.params.DISPLAY_SIZE[1]), 0, 32)
      pygame.display.set_caption('Billiard')
      self.clock = pygame.time.Clock()
    ## TODO REMOVE

  def _create_table(self):
    '''
    Creates the walls of the table
    :return:
    '''
    # Create walls in world RF
    left_wall_body = self.world.CreateStaticBody(position=(0, self.params.TABLE_CENTER[1]),
                                                 shapes=b2.b2PolygonShape(box=(self.params.WALL_THICKNESS/2,
                                                                               self.params.TABLE_SIZE[1]/2)))
    right_wall_body = self.world.CreateStaticBody(position=(self.params.TABLE_SIZE[0], self.params.TABLE_CENTER[1]),
                                                  shapes=b2.b2PolygonShape(box=(self.params.WALL_THICKNESS/2,
                                                                                self.params.TABLE_SIZE[1] / 2)))

    upper_wall_body = self.world.CreateStaticBody(position=(self.params.TABLE_CENTER[0], self.params.TABLE_SIZE[1]),
                                                  shapes=b2.b2PolygonShape(box=(self.params.TABLE_SIZE[0] / 2,
                                                                                self.params.WALL_THICKNESS/2)))
    bottom_wall_body = self.world.CreateStaticBody(position=(self.params.TABLE_CENTER[0], 0),
                                                   shapes=b2.b2PolygonShape(box=(self.params.TABLE_SIZE[0] / 2,
                                                                                 self.params.WALL_THICKNESS/2)))

    self.walls = [left_wall_body, upper_wall_body, right_wall_body, bottom_wall_body]

    # Create coordinate transform
    self.wt_transform = -self.params.TABLE_CENTER # world RF -> table RF
    self.tw_transform = self.params.TABLE_CENTER # table RF -> world RF

  def _create_balls(self, balls_pose):
    '''
    Creates the balls in the simulation at the given positions
    :param balls_pose: Initial pose of the ball in table RF
    :return:
    '''
    self.balls = []

    for pose in balls_pose:
      pose = pose + self.tw_transform # move balls in world RF
      ball = self.world.CreateDynamicBody(position=pose,
                                          bullet=True,
                                          allowSleep=True,
                                          fixtures=b2.b2FixtureDef(shape=b2.b2CircleShape(radius=self.params.BALL_RADIUS),
                                                                   density=1.0,
                                                                   friction=self.params.BALL_FRICTION,
                                                                   restitution=self.params.BALL_ELASTICITY))
      self.balls.append(ball)

  #TODO implement different intial arm positions
  def _create_robotarm(self, arm_position=None):
    '''
    Creates the robotic arm.
    :param angular_position: Initial angular position
    :return:
    '''
    link0 = self.world.CreateDynamicBody(position=(self.params.TABLE_CENTER[0], self.params.LINK_0_LENGTH/2),
                                         bullet=True,
                                         allowSleep=True,
                                         fixtures=b2.b2FixtureDef(
                                           shape=b2.b2PolygonShape(box=(self.params.LINK_THICKNESS,
                                                                        self.params.LINK_0_LENGTH/2)),
                                           density=1,
                                           friction=self.params.LINK_FRICTION,
                                           restitution=self.params.LINK_ELASTICITY))

    # The -.1 in the position is so that the two links can overlap in order to create the joint
    link1 = self.world.CreateDynamicBody(position=(self.params.TABLE_CENTER[0], self.params.LINK_0_LENGTH - .1 + self.params.LINK_1_LENGTH / 2),
                                         bullet=True,
                                         allowSleep=True,
                                         fixtures=b2.b2FixtureDef(
                                           shape=b2.b2PolygonShape(box=(self.params.LINK_THICKNESS,
                                                                        self.params.LINK_1_LENGTH / 2)),
                                           density=1,
                                           friction=self.params.LINK_FRICTION,
                                           restitution=self.params.LINK_ELASTICITY))

    jointW0 = self.world.CreateRevoluteJoint(bodyA=self.walls[3],
                                             bodyB=link0,
                                             anchor=self.walls[3].worldCenter,
                                             lowerAngle=-.5 * b2.b2_pi,
                                             upperAngle=.5 * b2.b2_pi,
                                             enableLimit=True,
                                             enableMotor=True)

    joint01 = self.world.CreateRevoluteJoint(bodyA=link0,
                                             bodyB=link1,
                                             anchor=link0.worldCenter + b2.b2Vec2((0, self.params.LINK_0_LENGTH/2)),
                                             lowerAngle=-b2.b2_pi,
                                             upperAngle=b2.b2_pi,
                                             enableLimit=False,
                                             maxMotorTorque=10.0,
                                             motorSpeed=0.0,
                                             enableMotor=True)

    self.arm = {'link0': link0, 'link1': link1, 'joint01': joint01, 'jointW0': jointW0}

  def _create_holes(self):
    '''
    Defines the holes in table RF. This ones are not simulated, but just defined as a list of dicts.
    :return:
    '''
    self.holes = [{'pose': np.array([-self.params.TABLE_SIZE[0] / 2, self.params.TABLE_SIZE[1] / 2]), 'radius': .4},
                  {'pose': np.array([self.params.TABLE_SIZE[0] / 2, self.params.TABLE_SIZE[1] / 2]), 'radius': .4}]

  ## TODO REMOVE
  def _clear_screen(self):
    """
    Clears the screen.
    :return: None
    """
    self.screen.fill(pygame.color.THECOLORS["white"])
  ## TODO REMOVE

  def reset(self, balls_pose, arm_position):
    # Destroy all the bodies
    for body in self.world.bodies:
      if body.type is b2.b2.dynamicBody:
        self.world.DestroyBody(body)

    # Recreate the balls and the arm
    self._create_balls(balls_pose)
    self._create_robotarm(arm_position)

  ## TODO REMOVE
  def _draw_world(self):
    """
    Draw the world.
    :return: None
    """
    # Draw holes. This are just drawn, but are not simulated.
    for hole in self.holes:
      pose = -hole['pose'] + self.tw_transform # To world transform (The - is to take into account pygame coordinate system)
      pygame.draw.circle(self.screen,
                         (255, 0, 0),
                         [int(pose[0] * self.params.PPM), int(pose[1] * self.params.PPM)],
                         int(hole['radius'] * self.params.PPM))

    for body in self.world.bodies:
      for fixture in body.fixtures:
        fixture.shape.draw(body, fixture, self.screen, self.params)
  ## TODO REMOVE

  def apply_torque_to_joint(self, joint, torque):
    self.arm[joint].motorSpeed = self.arm[joint].motorSpeed + torque * self.dt

  def step(self):
    '''
    Performs a simulator step
    :return:
    '''
    self.world.Step(self.dt, self.vel_iter, self.pos_iter)
    self.world.ClearForces()

    ## TODO REMOVE
    if self.render:
      self._clear_screen()
      self._draw_world()
      pygame.display.flip()
      self.clock.tick(self.params.TARGET_FPS)
    ## TODO REMOVE

if __name__ == "__main__":
  phys = PhysicsSim(balls_pose=[[0, 0], [1, 1]])
  for i in range(60):
    phys.apply_torque_to_joint('joint01', 1)
    phys.step()
    ball_pose = phys.balls[0].position
    dist = np.linalg.norm(ball_pose - phys.holes[0]['pose'])
    print(dist)