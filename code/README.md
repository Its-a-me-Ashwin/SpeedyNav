## Connect to XBOX controller via SSH
### Set up the dependencies
* sudo systemctl enable bluetooth
* sudo systemctl start bluetooth
* sudo apt update
* sudo apt install xboxdrv

### Connect
* bluetoothctl
* power on
* agent on
* default-agent
* scan on

Look for the MAC address of your XBOX controller. It will also display its name. 

* pair <MAC Address>
* trust <MAC Address>
* connect <MAC Address>
* exit

The controller would be connected to the RPi

## The project use special version of python with dependencies
* sudo ~/Projects/venv/bin/python
* Create an alias if necessary. 
