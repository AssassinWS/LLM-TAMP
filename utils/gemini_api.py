import google.generativeai as genai

from IPython.display import Markdown

from IPython.display import display

import textwrap

GOOGLE_API_KEY = ""

genai.configure(api_key=GOOGLE_API_KEY)

import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['HTTP_proxy'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_proxy'] = 'http://127.0.0.1:7890'

for m in genai.list_models():
  # if 'generateContent' in m.supported_generation_methods:
    print(m.name)

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

model = genai.GenerativeModel('gemini-pro')

ques = '''
You are an AI robot that generate a sequence of actions to reach the goal.

The tabletop environment has a robot arm, a basket and several boxes. The robot sits at (0, 0), faces positive x axis, while positive z axis points up.The basket has a square shape, ranges(0.55, 0.65) along the x axis, and ranges (-0.4, 0.2) along the y axis.
    The goal is to pack all the boxes into a compact basket on the tabletop.

    The robot has the following skills, where each skill can take a list of objects and parameters as input:
    - pick([obj], {}): pick up obj, with no parameters.
    - place([obj], {"x": [0.0, 1.0], "y": [-1.0, 1.0], "theta": [-3.14, 3.14]}): place obj at location (x, y) with planar rotation theta, where x ranges (0.0, 1.0), y ranges (-1.0, 1.0), and theta ranges (-3.14, 3.14).

    An example plan in json format that utilizes the above skills (assume there exists "red_box"):
    {
        "Plan": ["pick(['red_block'], {})", "place(['red_block'], {'x': 0.5, 'y': 0.0, 'theta': 0.0})", ...]
    }

Now the completed actions are: No past actions.
The current observation is: red_box is at position [0.4, 0.0, 0.05], red_box's bounding box corner is [0.3, -0.05, -0.0], [0.5, 0.05, 0.1], blue_box is at position [0.4, 0.3, 0.05], blue_box's bounding box corner is [0.35, 0.2, -0.0], [0.45, 0.4, 0.1], green_box is at position [0.4, -0.2, 0.05], green_box's bounding box corner is [0.35, -0.25, -0.0], [0.45, -0.15, 0.1] 
The current state is: red_box is not in basket, blue_box is not in basket, green_box is not in basket, 
Then compute a plan of actions to execute next (in json format):
'''
response = model.generate_content(ques)

print(response.text)
# print(to_markdown(response.text))