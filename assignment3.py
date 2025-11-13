import os

class td_qlearning:

  def __init__(self, directory):
    # directory is the path to a directory containing trials through state space

    self.possible_actions = [1, 2, 3] # number of coins to take from bag
    self.state_actions_each_trial = []
    self.explored_state_action_pairs = []
    self.alpha = 0.10
    self.gamma = 0.90
    self.threshold = 0.00000000000001 # minimum change needed to be observed to continue Q-value update iterations

    # store Q-function in an instance variable (a dictionary with key:(state, action) value:qvalue)
    self.qfunction = {}

    # learn Q-function from CSV file(s)
    for root, dirs, files in os.walk(directory):
      for file in files:
        if file.endswith(".csv"):
          file_path = os.path.join(root, file)
          with open(file_path, 'r') as f:
            state_actions = []
            for line in f:
              # skip empty lines
              line = line.strip()
              if not line:
                continue

              # parse state and action from line
              state_str, value_str = line.split(',')
              # store state in a tuple
              # (c_bag, c_agent, c_opponent, winner)
              s = tuple(state_str.split('/'))
              state = (int(s[0]), int(s[1]), int(s[2]), s[3])
              
              # convert action part to int, or keep as '-' for terminal state
              try:
                action = int(value_str)
              except ValueError:
                action = '-'
              state_actions.append((state, action))
            
            # add state-action pairs to a list of trials
            self.state_actions_each_trial.append(state_actions)
    
    # maintain a set of explored state-action pairs 
    self.explored_state_action_pairs = list(set([pair for sublist in self.state_actions_each_trial for pair in sublist]))
    
    # initilize all Q(s,a)
    for trial in self.state_actions_each_trial:
      for (state, action) in trial:          
          self.qfunction[(state, action)] = self.reward(state)

    # update Q-value functions until convergence
    max_diff = float('inf')
    num_trials = len(self.state_actions_each_trial)
    while max_diff >= self.threshold:
        max_diff = 0
        for j in range(num_trials):
          state_actions_pairs = self.state_actions_each_trial[j]
          for k in range(len(state_actions_pairs) - 1):
            prev_state = state_actions_pairs[k][0]
            prev_action = state_actions_pairs[k][1]
            current_state = state_actions_pairs[k + 1][0]
            diff = self.temporal_difference_qlearning(prev_state, prev_action, current_state)
            max_diff = max(abs(diff), max_diff)

  def temporal_difference_qlearning(self, prev_state, prev_action, current_state):
    # performs temporal difference qlearning
    # returns the error term for use in affirming convergence

    # consider possible actions
    action_qvalues = []
    for action in self.possible_actions:
      # if action is a terminal state, no action can be taken so use "-" action
      if (current_state[3] != '-'):
        action_qvalues.append(self.qvalue(current_state, "-"))
      # action is only valid if there are sufficient coins in the bag
      elif (current_state[0] - action) >= 0:
        action_qvalues.append(self.qvalue(current_state, action))
    
    # perform Q-value update
    if (action_qvalues):
      error_term = self.reward(prev_state) + self.gamma*max(action_qvalues) - self.qvalue(prev_state, prev_action)
      self.qfunction[prev_state, prev_action] += self.alpha*error_term

    return error_term

  def qvalue(self, state, action):
    # state is a string representation of a state
    # action is an integer representation of an action

    # find the qvalue of the given state action pair
    if ((state,action) in self.explored_state_action_pairs):
        return self.qfunction[state, action]
    else:
      # if not seen, qvalue defaults to would-be initial value reward(s)
      return self.reward(state)

  def policy(self, state):
    # state is a string representation of a state

    # terminal state, so no action can be taken
    if state[3] != '-':
      return 0

    # find the action that maximizes the qvalue
    best_action = self.possible_actions[0]
    for action in self.possible_actions[1:]:
      # only consider legal moves
      if (state[0] >= action):
        if self.qvalue(state, action) >= self.qvalue(state, best_action):
          best_action = action

    # Return the optimal action (as an integer) under the learned policy
    return best_action
  
  def reward(self,state):
    # helper function to calculate the reward value of a state
    # assumes state is a tuple (c_bag, c_agent, c_opponent, winner)

    match state[3]:
      case "A":
        return int(state[1])
      case "O":
        return -int(state[1])
      case _:
        return 0
      
# for testing the class
# if __name__ == "__main__":
#     learner = td_qlearning("Examples/Example2/Trials")  # folder containing CSVs
#     # print(len(td_qlearning.explored_state_action_pairs))
    
#     # print all learned Q-values
#     print("Learned Q-function:")
#     for key, value in learner.qfunction.items():
#         print(f"{key}: {value}")

#     # print("\nTest policy\n")

#     # example0
#     # print(learner.policy((11, 1, 1, '-')))

#     # example1
#     # print(learner.policy((6, 1, 6, '-')))
#     # print(learner.policy((0, 7, 6, 'O')))
#     # print(learner.policy((1, 8, 4, '-')))
#     # print(learner.policy((1, 6, 6, '-')))
#     # print(learner.policy((4, 5, 4, '-')))
#     # print(learner.policy((2, 8, 3, '-')))
#     # print(learner.policy((9, 2, 2, '-')))
#     # print(learner.policy((3, 6, 4, '-')))
#     # print(learner.policy((1, 5, 7, '-')))
#     # print(learner.policy((1, 9, 3, '-')))

#     # example2
#     # print(learner.policy((8, 3, 2, '-')))
#     # print(learner.policy((1, 8, 4, '-')))
#     # print(learner.policy((2, 8, 3, '-')))
#     # print(learner.policy((13, 0, 0, '-')))
#     # print(learner.policy((4, 4, 5, '-')))
#     # print(learner.policy((8, 2, 3, '-')))
#     # print(learner.policy((11, 0, 2, '-')))
#     # print(learner.policy((0, 3, 10, 'O')))
#     # print(learner.policy((10, 0, 3, '-')))
#     # print(learner.policy((5, 5, 3, '-')))

#     # example3
#     # print(learner.policy((1, 7, 5, '-')))
#     # print(learner.policy((9, 1, 3, '-')))
#     # print(learner.policy((4, 4, 5, '-')))  
#     # print(learner.policy((4, 3, 6, '-')))   
#     # print(learner.policy((6, 3, 4, '-')))   
#     # print(learner.policy((2, 4, 7, '-')))   
#     # print(learner.policy((1, 7, 5, '-')))   
#     # print(learner.policy((4, 6, 3, '-')))   
#     # print(learner.policy((13, 0, 0, '-')))  
#     # print(learner.policy((2, 4, 7, '-')))   

#     # print("\nTest qvalue\n")

#     # example0
#     # print(learner.qvalue((8,3,2,'-'),2))

#     # example1
#     # print(learner.qvalue((6, 1, 6, '-'), 2))
#     # print(learner.qvalue((0, 7, 6, 'O'), 0))
#     # print(learner.qvalue((1, 8, 4, '-'), 1))
#     # print(learner.qvalue((1, 6, 6, '-'), 1))
#     # print(learner.qvalue((4, 5, 4, '-'), 3))
#     # print(learner.qvalue((2, 8, 3, '-'), 2))
#     # print(learner.qvalue((9, 2, 2, '-'), 3))
#     # print(learner.qvalue((3, 6, 4, '-'), 2))
#     # print(learner.qvalue((1, 5, 7, '-'), 1))
#     # print(learner.qvalue((1, 9, 3, '-'), 1))

#     # example2
#     # print(learner.qvalue((8, 3, 2, '-'), 3))   
#     # print(learner.qvalue((1, 8, 4, '-'), 1))   
#     # print(learner.qvalue((2, 8, 3, '-'), 2))   
#     # print(learner.qvalue((13, 0, 0, '-'), 1))  
#     # print(learner.qvalue((4, 4, 5, '-'), 1))   
#     # print(learner.qvalue((8, 2, 3, '-'), 3))   
#     # print(learner.qvalue((11, 0, 2, '-'), 2))  
#     # print(learner.qvalue((0, 3, 10, 'O'), 0))  
#     # print(learner.qvalue((10, 0, 3, '-'), 3))  
#     # print(learner.qvalue((5, 5, 3, '-'), 1))   

#     # example3
#     # print(learner.qvalue((1, 7, 5, '-'), 1))   
#     # print(learner.qvalue((9, 1, 3, '-'), 3))   
#     # print(learner.qvalue((4, 4, 5, '-'), 2))   
#     # print(learner.qvalue((4, 3, 6, '-'), 1))   
#     # print(learner.qvalue((6, 3, 4, '-'), 3))   
#     # print(learner.qvalue((2, 4, 7, '-'), 1))   
#     # print(learner.qvalue((1, 7, 5, '-'), 1))   
#     # print(learner.qvalue((4, 6, 3, '-'), 2))   
#     # print(learner.qvalue((13, 0, 0, '-'), 2))  
#     # print(learner.qvalue((2, 4, 7, '-'), 2))