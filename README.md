# Speedy Navigation in Closed Non-Static Environment
## Master Project by Ashwin R Bharadwaj

This project focuses on developing efficient navigation techniques for a mobile robot operating in a dynamic, non-static environment. The aim is to achieve real-time performance in decision-making and navigation while minimizing computational overhead.

### Hardware
For a detailed overview of the hardware utilized in this project, please refer to the following repository: [UniBot Hardware](https://github.com/Its-a-me-Ashwin/UniBot.git).

### Software
- **Image Embedding**: Evaluated two pre-trained neural network architectures, VGG16 and Xception, for converting captured images into embeddings. The final choice will depend on comparative performance tests.
- **Depth Image Processing**: Implemented resolution reduction and flattening techniques for the depth images to optimize computational efficiency.
- **POMDP Development**: A Partially Observable Markov Decision Process (POMDP) has been designed to accurately model the robot's environment, factoring in uncertainties in both observations and actions.

### TODO
- **GUI Development**: Design and implement a graphical user interface (GUI) to parse and display telemetry data from the robot.
- **Telemetry Setup**: Establish a telemetry link between the robot and a Raspberry Pi to transmit real-time sensor data.
- **MDP Testing**: Further validate the Markov Decision Process (MDP) in various real-world and simulated scenarios.
- **Hardware Enhancements**: Print a top mount for the robot and attach sensors at adjustable angles to improve the range and accuracy of environmental perception.

---

For additional details or collaboration inquiries, please feel free to reach out.
