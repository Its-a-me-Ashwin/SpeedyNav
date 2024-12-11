import os
from typing import Optional, List

class Node:
    def __init__(self, node_id: str, x: int, y: int):
        self.node_id = node_id  # Unique identifier for the node
        self.x = x  # X-coordinate
        self.y = y  # Y-coordinate

        # Paths for different orientations
        self.image_paths = {"NORTH": None, "EAST": None, "SOUTH": None, "WEST": None}
        self.depth_paths = {"NORTH": None, "EAST": None, "SOUTH": None, "WEST": None}
        self.embedding_paths = {"NORTH": None, "EAST": None, "SOUTH": None, "WEST": None}

        # Connections to other nodes
        self.forward: Optional[Node] = None
        self.back: Optional[Node] = None
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None

        # Tag attribute for additional information
        self.tag = {}

    def set_connection(self, direction: str, node: Optional['Node']):
        """Set the connection for a given direction."""
        if direction not in {"forward", "back", "left", "right"}:
            raise ValueError(f"Invalid direction: {direction}. Choose from 'forward', 'back', 'left', 'right'.")
        setattr(self, direction, node)

    def set_data_paths(self, orientation: str, image_path: str, depth_path: str, embedding_path: str):
        """Set paths for image, depth, and embedding data for a specific orientation."""
        if orientation not in {"NORTH", "EAST", "SOUTH", "WEST"}:
            raise ValueError(f"Invalid orientation: {orientation}. Choose from 'NORTH', 'EAST', 'SOUTH', 'WEST'.")
        self.image_paths[orientation] = image_path
        self.depth_paths[orientation] = depth_path
        self.embedding_paths[orientation] = embedding_path

class Graph:
    def __init__(self, graph_id: str):
        self.graph_id = graph_id  # Unique identifier for the graph
        self.nodes = []  # List of all nodes in the graph
        self.path = f"./Graphs_{self.graph_id}"  # Base path for all graph data

        # Create the base directory for the graph if it doesn't exist
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def add_node(self, node: Node):
        """Add a node to the graph and organize its data under the graph path."""
        node_dir = os.path.join(self.path, f"node_{node.node_id}")

        # Create a directory for the node if it doesn't exist
        if not os.path.exists(node_dir):
            os.makedirs(node_dir)

        # Update paths to ensure they're within the graph structure
        for orientation in ["NORTH", "EAST", "SOUTH", "WEST"]:
            if node.image_paths[orientation]:
                node.image_paths[orientation] = os.path.join(node_dir, os.path.basename(node.image_paths[orientation]))
            if node.depth_paths[orientation]:
                node.depth_paths[orientation] = os.path.join(node_dir, os.path.basename(node.depth_paths[orientation]))
            if node.embedding_paths[orientation]:
                node.embedding_paths[orientation] = os.path.join(node_dir, os.path.basename(node.embedding_paths[orientation]))

        self.nodes.append(node)

    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Retrieve a node by its ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def save_node_data(self, node: Node, orientation: str, image_data, depth_data, embedding_data):
        """Save the image, depth, and embedding data for a node for a specific orientation."""
        if orientation not in {"NORTH", "EAST", "SOUTH", "WEST"}:
            raise ValueError(f"Invalid orientation: {orientation}. Choose from 'NORTH', 'EAST', 'SOUTH', 'WEST'.")

        if node.image_paths[orientation] and not os.path.exists(node.image_paths[orientation]):
            with open(node.image_paths[orientation], "wb") as f:
                f.write(image_data)
        if node.depth_paths[orientation] and not os.path.exists(node.depth_paths[orientation]):
            with open(node.depth_paths[orientation], "wb") as f:
                f.write(depth_data)
        if node.embedding_paths[orientation] and not os.path.exists(node.embedding_paths[orientation]):
            with open(node.embedding_paths[orientation], "wb") as f:
                f.write(embedding_data)

    def save_data_to_node(self, node: Node, orientation: str, image_data, depth_data, embedding_data):
        """Save data to a node for a specific orientation and ensure paths are set correctly."""
        node_dir = os.path.join(self.path, f"node_{node.node_id}")
        if not os.path.exists(node_dir):
            os.makedirs(node_dir)

        image_path = os.path.join(node_dir, f"image_{orientation.lower()}.png")
        depth_path = os.path.join(node_dir, f"depth_{orientation.lower()}.npy")
        embedding_path = os.path.join(node_dir, f"embedding_{orientation.lower()}.npy")

        node.set_data_paths(orientation, image_path, depth_path, embedding_path)

        with open(image_path, "wb") as f:
            f.write(image_data)
        with open(depth_path, "wb") as f:
            f.write(depth_data)
        with open(embedding_path, "wb") as f:
            f.write(embedding_data)

    def get_adjacent_nodes(self, node: Node) -> List[Node]:
        """Get all directly connected nodes of a given node."""
        connections = [
            node.forward, node.back, node.left, node.right
        ]
        return [conn for conn in connections if conn is not None]

    def get_nodes_at_distance(self, start_node: Node, distance: int) -> List[Node]:
        """Get all nodes at a specified distance from the start node."""
        visited = set()
        current_level = [start_node]
        visited.add(start_node.node_id)

        for _ in range(distance):
            next_level = []
            for node in current_level:
                for neighbor in self.get_adjacent_nodes(node):
                    if neighbor.node_id not in visited:
                        visited.add(neighbor.node_id)
                        next_level.append(neighbor)
            current_level = next_level

        return current_level

# Example usage
if __name__ == "__main__":
    # Create a graph
    graph = Graph("B")

    # Create a node
    node_a = Node(node_id="A", x=0, y=0)

    # Set data paths for each orientation
    node_a.set_data_paths("NORTH", "image_1_north.png", "depth_1_north.npy", "embedding_1_north.npy")
    node_a.set_data_paths("EAST", "image_1_east.png", "depth_1_east.npy", "embedding_1_east.npy")
    node_a.set_data_paths("SOUTH", "image_1_south.png", "depth_1_south.npy", "embedding_1_south.npy")
    node_a.set_data_paths("WEST", "image_1_west.png", "depth_1_west.npy", "embedding_1_west.npy")

    # Add the node to the graph
    graph.add_node(node_a)

    # Save some dummy data for the node in different orientations
    graph.save_node_data(node_a, "NORTH", b"dummy_image_data_north", b"dummy_depth_data_north", b"dummy_embedding_data_north")
    graph.save_node_data(node_a, "EAST", b"dummy_image_data_east", b"dummy_depth_data_east", b"dummy_embedding_data_east")

    # Use the new method to save data to a node
    graph.save_data_to_node(node_a, "SOUTH", b"new_dummy_image_data_south", b"new_dummy_depth_data_south", b"new_dummy_embedding_data_south")

    # Access and modify node connections
    node_b = Node(node_id="B", x=1, y=0)
    node_b.set_data_paths("NORTH", "image_2_north.png", "depth_2_north.npy", "embedding_2_north.npy")
    graph.add_node(node_b)
    node_a.set_connection("forward", node_b)

    node_c = Node(node_id="C", x=1, y=1)
    node_c.set_data_paths("NORTH", "image_3_north.png", "depth_3_north.npy", "embedding_3_north.npy")
    graph.add_node(node_c)
    node_b.set_connection("right", node_c)

    node_d = Node(node_id="D", x=0, y=1)
    node_d.set_data_paths("NORTH", "image_4_north.png", "depth_4_north.npy", "embedding_4_north.npy")
    graph.add_node(node_d)
    node_a.set_connection("right", node_d)
    node_d.set_connection("back", node_c)

    node_e = Node(node_id="E", x=-1, y=0)
    node_e.set_data_paths("NORTH", "image_5_north.png", "depth_5_north.npy", "embedding_5_north.npy")
    graph.add_node(node_e)
    node_a.set_connection("left", node_e)

    # Find adjacent nodes
    adjacent_nodes = graph.get_adjacent_nodes(node_a)
    print(f"Node A adjacent nodes: {[node.node_id for node in adjacent_nodes]}")

    # Find nodes at distance 2 from node A
    nodes_at_distance = graph.get_nodes_at_distance(node_a, 2)
    print(f"Nodes at distance 2 from Node A: {[node.node_id for node in nodes_at_distance]}")
