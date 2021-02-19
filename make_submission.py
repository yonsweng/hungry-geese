from kaggle_environments import make

env = make("hungry_geese", debug=True) #set debug to True to see agent internals each step

env.run(["submission.py", "submission.py", "submission.py", "submission.py"])
# env.render(mode="ipython", width=800, height=700)