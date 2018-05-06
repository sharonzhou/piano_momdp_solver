import xml.etree.cElementTree as ET

tree = ET.parse('piano.pomdpx')
root = tree.getroot()
print(root)
print(root.tag, root.attrib)

for child in root:
	print("child", child, child.tag, child.attrib)
	for grandchild in child:
		print("grandchild", grandchild, grandchild.tag, grandchild.attrib)


root = ET.Element("pomdpx", {"xsi:noNamespaceSchemaLocation": "pomdpx.xsd", "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance"}, version="0.1", id="tagxmlfac")
ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")

"""Text description"""
description = ET.SubElement(root, "Description").text = "written by Sharon Zhou sharonz@cs.stanford.edu\n\n#Total number of states in curriculum: 16\n#Total number of target states: 1"

"""Discount factor"""
discount = ET.SubElement(root, "Discount").text = "0.95"

"""Nodes/variables in graph"""
variable = ET.SubElement(root, "Variable")

# Iterate over states (curriculum graph) TODO
A = ET.SubElement(variable, "StateVar", vnamePrev="A_0", vnameCurr="A", fullyObs="true")
values = ET.SubElement(A, "ValueEnum").text = "0 1"

# Observable variables (performance, duration on task, exercised autonomy, exercised difficulty) TODO Values
performance = ET.SubElement(variable, "ObsVar", vname="performance", fullyObs="true")
duration = ET.SubElement(variable, "ObsVar", vname="duration", fullyObs="true")
exercised_autonomy = ET.SubElement(variable, "ObsVar", vname="exercised_autonomy", fullyObs="true")
exercised_difficulty = ET.SubElement(variable, "ObsVar", vname="exercised_difficulty", fullyObs="true")

# Partially observable variables (desired autonomy, desired difficulty) TODO values
desired_autonomy = ET.SubElement(variable, "ObsVar", vname="desired_autonomy", fullyObs="false")
desired_difficulty = ET.SubElement(variable, "ObsVar", vname="desired_difficulty", fullyObs="false")

# Action space/action variables (give autonomy, give difficulty)
give_autonomy = ET.SubElement(variable, "ActionVar", vname="give_autonomy")
give_difficulty = ET.SubElement(variable, "ActionVar", vname="give_difficulty")

# Reward variable?
# reward = ET.SubElement(variable, "RewardVar", vname="reward")

"""Initial state belief"""
initialState = ET.SubElement(root, "InitialStateBelief").text = "0.95"

"""Transition functions"""
transition = ET.SubElement(root, "StateTransitionFunction").text = "0.95"

"""Observation function"""
observation = ET.SubElement(root, "ObsFunction").text = "0.95"

"""Reward function"""
reward = ET.SubElement(root, "RewardFunction").text = "0.95"

tree = ET.ElementTree(root)
tree.write("piano_auto.pomdpx", xml_declaration=True, encoding="ISO-8859-1")
