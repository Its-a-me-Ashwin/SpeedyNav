import numpy as np
import json
from datetime import datetime
from skimage.restoration import denoise_tv_chambolle
import argparse
from cnn import EfficientNetEdgeNet, get_image_embedding
from graph import Node, Graph
from realsense import Camera
from odrive import RobotController

# Define directional and action constants
directions = ["NORTH", "WEST", "SOUTH", "EAST"]
actions = ["RIGHT", "LEFT", "FORWARD", "BACKWARD"]

class Orientation(object):
    """
    Class to handle the orientation of the robot and its movements.
    Tracks the current position and direction the robot is facing.
    """
    def __init__(self):
        self.currentX = 0  # Current X-coordinate
        self.currentY = 0  # Current Y-coordinate
        self.direction = 0  # 0: NORTH, 1: EAST, 2: SOUTH, 3: WEST
        self.cameraDirection = 0  # Tracks the camera's facing direction

    def changeCameraDirection(self, action):
        """
        Update the camera's facing direction based on the given action.
        """
        if action == actions[0]:  # RIGHT
            self.cameraDirection = (self.cameraDirection + 1) % 4
        elif action == actions[1]:  # LEFT
            self.cameraDirection = (self.cameraDirection - 1) % 4
        else:
            print("Invalid action")

    def changeState(self, action):
        """
        Change the robot's position or direction based on the given action.
        """
        if action == actions[0]:  # Turn right
            self.direction = (self.direction + 1) % 4
        elif action == actions[1]:  # Turn left
            self.direction = (self.direction - 1) % 4
        elif action == actions[2]:  # Move forward
            if self.direction == 0:  # Facing NORTH
                self.currentY += 1
            elif self.direction == 1:  # Facing EAST
                self.currentX += 1
            elif self.direction == 2:  # Facing SOUTH
                self.currentY -= 1
            elif self.direction == 3:  # Facing WEST
                self.currentX -= 1
        elif action == actions[3]:  # Move backward
            if self.direction == 0:  # Facing NORTH
                self.currentY -= 1
            elif self.direction == 1:  # Facing EAST
                self.currentX -= 1
            elif self.direction == 2:  # Facing SOUTH
                self.currentY += 1
            elif self.direction == 3:  # Facing WEST
                self.currentX += 1
        else:
            print("Invalid action")

def exploreNode(graph, orientation, current_node, camera, model, robot, armUse):
    """
    Capture data for the current node in all four directions and save embeddings.
    """
    if not armUse:
        for _ in range(4):  # Loop through all four directions
            # Capture color and depth frames and invert them
            color_avg, depth_avg = camera.long_exposure(1)
            color_avg, depth_avg = Camera.invert_frame(color_avg, depth_avg)

            # Generate embeddings from the captured image
            embedding = get_image_embedding(model, color_avg, "cpu")
            if embedding is not None:
                graph.save_data_to_node(
                    current_node,
                    directions[orientation.cameraDirection],
                    color_avg,
                    depth_avg,
                    embedding
                )

            # Turn the robot 90 degrees and update camera orientation
            robot.turn(90)
            orientation.changeCameraDirection(actions[0])

def moveRobotOnAction(action, robot):
    """
    Execute robot movements based on the specified action.
    """
    if action == "forward":
        robot.moveLinear(100)
    elif action == "back":
        robot.turn(180)
        robot.moveLinear(100)
    elif action == "left":
        robot.turn(90)
        robot.moveLinear(100)
    elif action == "right":
        robot.turn(-90)
        robot.moveLinear(100)

def explore_with_dfs(graph, camera, orientation, model, robot, armUse):
    """
    Perform Depth-First Search (DFS) to explore the environment and build the graph.
    """
    directions_map = {"NORTH": "forward", "SOUTH": "back", "EAST": "right", "WEST": "left"}
    stack = [(orientation.currentX, orientation.currentY)]  # Stack for DFS
    visited = set()  # Track visited nodes

    while stack:
        current_x, current_y = stack.pop()

        # Skip if the node has already been visited
        if (current_x, current_y) in visited:
            continue
        visited.add((current_x, current_y))

        node_id = f"node_{current_x}_{current_y}"
        print(f"Exploring: {node_id}")

        # Add current node to graph if it doesn't already exist
        current_node = graph.get_node_by_id(node_id)
        if not current_node:
            current_node = Node(node_id=node_id, x=current_x, y=current_y)
            graph.add_node(current_node)

        # Capture data for the current node
        exploreNode(graph, orientation, current_node, camera, model, robot, armUse)

        # Explore neighbors
        for action, movement in zip(actions, ["forward", "left", "back", "right"]):
            orientation.changeState(action)  # Change orientation based on action
            neighbor_x, neighbor_y = orientation.currentX, orientation.currentY
            neighbor_id = f"node_{neighbor_x}_{neighbor_y}"

            if (neighbor_x, neighbor_y) not in visited and not camera.isWallPresent():
                # Move the robot in the specified direction
                moveRobotOnAction(action, robot)

                # Add the neighbor to the stack
                stack.append((neighbor_x, neighbor_y))

                # Add the neighbor node to the graph if it doesn't exist
                neighbor_node = graph.get_node_by_id(neighbor_id)
                if not neighbor_node:
                    neighbor_node = Node(node_id=neighbor_id, x=neighbor_x, y=neighbor_y)
                    graph.add_node(neighbor_node)

                # Set bidirectional connections between nodes
                current_node.set_connection(movement, neighbor_node)
                neighbor_node.set_connection({"forward": "back", "back": "forward", "left": "right", "right": "left"}[movement], current_node)

            # Reverse the orientation change after checking
            orientation.changeState(actions[3])

def explore():
    """
    Main function to set up the environment and initiate exploration.
    """
    parser = argparse.ArgumentParser(description="Controller")
    parser.add_argument("--name", type=str, default="test", help="Name of the map to create")
    parser.add_argument("--user_robot_arm", type=int, default=0, help="Set to 0 to disable robot arm, 1 to enable")
    args = parser.parse_args()

    graphName = args.name if args.name else "test"
    armUse = args.user_robot_arm

    # Load camera configuration
    with open('../realsense/d435.config', 'r') as f:
        config = json.load(f)

    def custom_filters(x):
        pass

    device = 'cpu'
    camera = Camera(config=config, filters=custom_filters)
    model = EfficientNetEdgeNet().to(device)
    robot = RobotController(maxSpeed=5)
    robot.motorController.setClosedLoopControl(0)
    robot.motorController.setClosedLoopControl(1)
    graph = Graph(graphName)
    currentOrientation = Orientation()

    # Start the exploration process using DFS
    explore_with_dfs(graph, camera, currentOrientation, model, robot, armUse)
