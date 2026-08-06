
from .node import ANode, ONode, find_new_PO_root, \
    particle_revigoration
import random
import time
from .estimation_stub import type_parameter_estimation

class POMCP(object):

    def __init__(self,max_depth,max_it,kwargs):
        ###
        # Traditional Monte-Carlo Tree Search parameters
        ###
        self.root = None
        self.max_depth = max_depth
        self.max_it = max_it
        exploration_constant = kwargs.get('exploration_constant')
        self.c = float(exploration_constant) if exploration_constant is not None else 0.5
        reward_scale = kwargs.get('reward_scale')
        self.reward_scale = float(reward_scale) if reward_scale is not None else 1.0
        if self.c < 0.0:
            raise ValueError('exploration_constant must be nonnegative')
        if self.reward_scale <= 0.0:
            raise ValueError('reward_scale must be positive')
        root_min_action_visits = kwargs.get('root_min_action_visits')
        self.root_min_action_visits = (
            int(root_min_action_visits)
            if root_min_action_visits is not None
            else 1
        )
        if self.root_min_action_visits < 1:
            raise ValueError('root_min_action_visits must be at least one')
        discount_factor = kwargs.get('discount_factor')
        self.discount_factor = discount_factor\
            if discount_factor is not None else 0.95

        ###
        # POMCP enhancements
        ###
        # particle Revigoration (silver2010pomcp)
        particle_revigoration = kwargs.get('particle_revigoration')
        if particle_revigoration is not None:
            self.pr = particle_revigoration
        else: #default
            self.pr = True

        k = kwargs.get('k') # particle filter size
        self.k = k if k is not None else 100

        ###
        # Further settings
        ###
        target = kwargs.get('target')
        if target is not None:
            self.target = target
            self.initial_target = target
        else: #default
            self.target = 'max'
            self.initial_target = 'max'
            
        adversary_mode = kwargs.get('adversary')
        if adversary_mode is not None:
            self.adversary = adversary_mode
        else: #default
            self.adversary = False
            
        stack_size = kwargs.get('state_stack_size')
        if stack_size is not None:
            self.state_stack_size = stack_size
        else: #default
            self.state_stack_size = 1

        ###
        # Evaluation
        ###
        self.rollout_total_time = 0.0
        self.rollout_count = 0.0
        
        self.simulation_total_time = 0.0
        self.simulation_count = 0.0

    def change_paradigm(self):
        if self.target == 'max':
            return 'min'
        elif self.target == 'min':
            return 'max'
        else:
            raise NotImplemented

    def simulate_action(self, node, action):
        # 1. Copying the current state for simulation
        tmp_state = node.state.copy()

        # 2. Acting
        next_state,reward, _, _ = tmp_state.step(action)
        # O -> A is an action edge and does not advance environment time.
        # The following A -> O edge advances depth by exactly one step.
        next_node = ANode(action,next_state,node.depth,node)

        # 3. Returning the next node and the reward
        return next_node, reward

    def rollout_policy(self,state):
        if getattr(state,'default_policy',None) is not None:
            return state.default_policy()
        return random.choice(state.get_actions_list())

    def rollout(self,node):
        # 1. Checking if it is an end state or leaf node
        if self.is_terminal(node) or self.is_leaf(node):
            return 0

        self.rollout_count += 1
        start_t = time.time()

        # Optional deterministic leaf bootstrap. Tree search and root action
        # selection remain POMCP; this replaces only a high-variance random
        # rollout from a newly expanded leaf.
        rollout_value = getattr(node.state, 'rollout_value', None)
        if callable(rollout_value):
            value = float(rollout_value())
            self.rollout_total_time += (time.time() - start_t)
            return value

        # 2. Choosing an action
        action = self.rollout_policy(node.state)

        # 3. Simulating the action
        next_state, reward, _, _ = node.state.step(action)
        node.state = next_state
        node.observation = next_state.get_observation()
        node.depth += 1

        end_t = time.time()
        self.rollout_total_time += (end_t - start_t)

        # 4. Rolling out
        return reward +\
            self.discount_factor*self.rollout(node)

    def get_rollout_node(self,node):
        obs = node.state.get_observation()
        tmp_state = node.state.copy()
        depth = node.depth
        return ONode(observation=obs,state=tmp_state,depth=depth,parent=None)

    def is_leaf(self, node):
        if node.depth >= self.max_depth:
            return True
        return False

    def is_terminal(self, node):
        return node.state.state_set.is_final_state(node.state)

    def simulate(self, node):
        # 1. Checking the stop condition
        if node.depth == 0:
            node.visits += 1

        if self.is_terminal(node) or self.is_leaf(node):
            return 0

        # 2. Checking child nodes
        if node.children == []:
            # a. adding the children
            for action in node.actions:
                (next_node, reward) = self.simulate_action(node, action)
                node.children.append(next_node)
            rollout_node = self.get_rollout_node(node)
            return self.rollout(rollout_node)

        self.simulation_count += 1
        start_t = time.time()
        
        # 3. Selecting the best action
        under_sampled = []
        if node.depth == 0:
            under_sampled = [
                candidate
                for candidate in node.actions
                if int(node.qtable[str(candidate)]['trials'])
                < self.root_min_action_visits
            ]
        if under_sampled:
            minimum_visits = min(
                int(node.qtable[str(candidate)]['trials'])
                for candidate in under_sampled
            )
            least_sampled = [
                candidate
                for candidate in under_sampled
                if int(node.qtable[str(candidate)]['trials']) == minimum_visits
            ]
            action = random.choice(least_sampled)
        else:
            action = node.select_action(
                coef=self.c * self.reward_scale,
                mode=self.target,
            )
        self.target = self.change_paradigm() if self.adversary else self.target   

        # 4. Simulating the action
        (action_node, reward) = self.simulate_action(node, action)

        # 5. Adding the action child on the tree
        if action_node.action in [c.action for c in node.children]:
            for child in node.children:
                if action_node.action == child.action:
                    child.state = action_node.state.copy()
                    action_node = child
                    break
        else:
            node.children.append(action_node)
        action_node.visits += 1

        # 6. Getting the observation and adding the observation child on the tree
        observation_node = None
        observation = action_node.state.get_observation()
        
        for child in action_node.children:
            if child.state.observation_is_equal(observation):
                observation_node = child
                observation_node.state = action_node.state.copy()
                observation_node.particle_filter.append(action_node.state)
                break
        
        if observation_node is None:
            observation_node = action_node.add_child(observation)
            observation_node.particle_filter.append(observation_node.state)
        observation_node.visits += 1

        end_t = time.time()
        self.simulation_total_time += (end_t - start_t)

        # 7. Calculating the reward, quality and updating the node
        R = reward + float(self.discount_factor * self.simulate(observation_node))
        if node.depth > 0:
            node.particle_filter.append(node.state)
        node.update(action, R)
        return R

    def search(self, node, agent):
        # 1. Performing the Monte-Carlo Tree Search
        it = 0
        root_belief_state = node.state.copy()
        stratified = getattr(root_belief_state, 'stratified_sample_states', None)
        root_schedule = (
            list(stratified(self.max_it))
            if callable(stratified)
            else None
        )
        while it < self.max_it:
            self.target = self.initial_target

            # a. Independently sample the root hidden state from the exact
            # planner-facing belief.  Reusing and appending sampled root
            # particles creates a Polya-urn drift away from that belief.
            beliefState = (
                root_schedule[it].copy()
                if root_schedule is not None
                else root_belief_state.sample_state(agent)
            )
            node.state = beliefState

            # b. simulating
            self.simulate(node)

            it += 1

        node.state = root_belief_state
        self.target = self.initial_target
        return node.get_best_action(self.target)

    def planning(self, state, agent):
        # 1. Getting the current state and previous action-observation pair
        previous_action = agent.next_action
        current_observation = state.get_observation()

        # 2. Defining the root of our search tree
        # via initialising the tree
        if self.root is None:
            self.root = ONode(observation=None,state=state,depth=0,parent=None)
        # or advancing within the existent tree
        else:
            self.root = find_new_PO_root(state, previous_action,\
             current_observation, agent, self.root, adversary=self.adversary)
        
        # 3. Estimating the parameters 
        if 'estimation_method' in agent.smart_parameters:
            self.root.state, agent.smart_parameters['estimation'] = \
             type_parameter_estimation(self.root.state,agent, agent.smart_parameters\
              ['estimation_method'], *agent.smart_parameters['estimation_args'])

        # 4. Performing particle revigoration
        if self.pr:
            particle_revigoration(state,agent,self.root,self.k)

        # 5. Searching for the best action within the tree
        best_action = self.search(self.root, agent)

        # 6. Returning the best action
        # NOTE: 禁止在基准评测中打印 Q-table（用户要求输出保持干净）
        # self.root.show_qtable()
        root_action_stats = {}
        for action in self.root.actions:
            row = self.root.qtable[str(action)]
            action_name = getattr(action, 'name', str(action))
            root_action_stats[action_name] = {
                'action_value': int(action) if hasattr(action, '__int__') else str(action),
                'q_value': float(row['qvalue']),
                'sum_value': float(row['sumvalue']),
                'visits': int(row['trials']),
            }
        info = {
            'nrollouts': self.rollout_count,
            'nsimulations': self.simulation_count,
            'root_visits': int(self.root.visits),
            'root_action_stats': root_action_stats,
            'ucb_exploration_constant': float(self.c),
            'ucb_reward_scale': float(self.reward_scale),
            'ucb_effective_coefficient': float(self.c * self.reward_scale),
            'root_min_action_visits': int(self.root_min_action_visits),
        }
        return best_action, info

def pomcp_planning(env, agent, max_depth=20, max_it=250, **kwargs):    
    # 1. Setting the environment for simulation
    copy_env = env.copy()
    copy_env.simulation = True

    # 2. POMCP Planning
    # - initialising/getting the plannin algorithm
    pomcp = POMCP(max_depth, max_it, kwargs) if 'pomcp' not \
     in agent.smart_parameters else agent.smart_parameters['pomcp']
    
    # - planning
    t0 = time.time()
    next_action, info = pomcp.planning(copy_env,agent)
    info = dict(info or {})
    info["time_s"] = float(time.time() - t0)

    # 3. Updating the search tree
    agent.smart_parameters['pomcp'] = pomcp
    agent.smart_parameters['count'] = info
    return next_action, info

# from .node import ANode, ONode, find_new_PO_root, \
#     particle_revigoration
# import random
# import time

# #  ONode(h)：表示一个 history 末尾是 observation 的节点, 它持有一个 particle_filter：近似 b(h)
# #  ANode(h,a)：表示在某个 history 下采取某动作后的节点.  ONode -- ANode -- ONode -- ...

# class POMCP:
#     def __init__(self, max_depth: int, max_it: int, kwargs):
#         self.root = None
#         self.max_depth = max_depth
#         self.max_it = max_it
#         self.c = 0.5
#         discount_factor = kwargs.get('discount_factor')
#         self.discount_factor = discount_factor\
#             if discount_factor is not None else 0.95
#         k = kwargs.get('k') # particle filter size
#         self.k = k if k is not None else 100
#         self.target = 'max'  # 没有对手博弈的过程，默认选择最大值
#         particle_revigoration = kwargs.get('particle_revigoration')
#         if particle_revigoration is not None:
#             self.pr = particle_revigoration
#         else: # 默认是进行 reinvigoration
#             self.pr = True

#         self.adversary =  False  # 没有对手博弈的过程，默认是 False

#         # Evaluation counters (initialized here; reset per planning call)
#         self.rollout_total_time = 0.0
#         self.rollout_count = 0
#         self.simulation_total_time = 0.0
#         self.simulation_count = 0

#     def is_leaf(self, node):
#         # Depth semantics: O-nodes count *environment steps / actions* from root.
#         if node.depth >= self.max_depth:
#             return True
#         return False

#     def is_terminal(self, node):
#         return node.state.state_set.is_final_state(node.state)

#     def simulate_action(self, node, action):
#         # 1. Copying the current state for simulation
#         tmp_state = node.state.copy()

#         # 2. Acting
#         next_state,reward, _, _ = tmp_state.step(action)
#         # Depth semantics: O -> A does NOT advance time; A -> O advances by +1 (see node.py).
#         next_node = ANode(action,next_state,node.depth,node)

#         # 3. Returning the next node and the reward
#         return next_node, reward

#     def rollout_policy(self,state):
#         if getattr(state,'default_policy',None) is not None:
#             return state.default_policy()
#         return random.choice(state.get_actions_list())

#     def get_rollout_node(self,node):  #  rollout 会污染节点
#         obs = node.state.get_observation()
#         tmp_state = node.state.copy()
#         depth = node.depth
#         return ONode(observation=obs,state=tmp_state,depth=depth,parent=None)

#     def rollout(self,node):
#         # 1. Checking if it is an end state or leaf node
#         if self.is_terminal(node) or self.is_leaf(node):
#             return 0

#         self.rollout_count += 1
#         start_t = time.time()

#         # 2. Choosing an action
#         action = self.rollout_policy(node.state)

#         # 3. Simulating the action
#         next_state, reward, _, _ = node.state.step(action)
#         node.state = next_state
#         node.observation = next_state.get_observation()
#         # One environment step simulated.
#         node.depth += 1

#         end_t = time.time()
#         self.rollout_total_time += (end_t - start_t)

#         # 4. Rolling out
#         return reward +\
#             self.discount_factor*self.rollout(node)  # 递归

#     def simulate(self, node):
#         # 1. Checking the stop condition
#         if node.depth == 0:
#             node.visits += 1

#         if self.is_terminal(node) or self.is_leaf(node):
#             return 0

#         # 2. Checking child nodes
#         if node.children == []:
#             # 到达一个未扩展的 history 节点 h，把所有动作子节点挂上，但不继续深入树，而是直接 rollout 估值。
#             for action in node.actions:
#                 (next_node, reward) = self.simulate_action(node, action)
#                 node.children.append(next_node)

#             rollout_node = self.get_rollout_node(node)
#             return self.rollout(rollout_node)
#         self.simulation_count += 1
#         start_t = time.time()

#         # 3.  UCT 动作选择
#         action = node.select_action(coef=self.c,mode=self.target)

#         # 4.  生成动作后继
#         (action_node, reward) = self.simulate_action(node, action)

#         # 5. 把该 action child 合并到树里（避免重复 action 节点）
#         if action_node.action in [c.action for c in node.children]:
#             for child in node.children:
#                 if action_node.action == child.action:
#                     child.state = action_node.state.copy()
#                     action_node = child
#                     break
#         else:
#             node.children.append(action_node)
#         action_node.visits += 1

#         # 6. 按 observation 分裂形成子树
#         observation_node = None
#         observation = action_node.state.get_observation()
        
#         for child in action_node.children:
#             if child.state.observation_is_equal(observation):  # 同样观测落在同一个 ONode
#                 observation_node = child
#                 observation_node.state = action_node.state.copy()
#                 observation_node.particle_filter.append(action_node.state) # 模拟得到的 𝑠′ 作为一个粒子加入到 b(h′)
#                 break
        
#         if observation_node is None:
#             observation_node = action_node.add_child(observation)
#             observation_node.particle_filter.append(observation_node.state)
#         observation_node.visits += 1

#         end_t = time.time()
#         self.simulation_total_time += (end_t - start_t)

#         # 7. 递归模拟 + 回传（backup）
#         R = reward + float(self.discount_factor * self.simulate(observation_node))
#         node.particle_filter.append(node.state)
#         node.update(action, R) # 对节点访问数， 节点值等做更新
#         return R        


#     def search(self, node, agent):
#         # 1. Performing the Monte-Carlo Tree Search
#         it = 0
#         while it < self.max_it:
#             # a. Sampling the belief state for simulation
#             if len(node.particle_filter) == 0:
#                 beliefState = node.state.sample_state(agent)
#             else:
#                 beliefState = random.sample(node.particle_filter,1)[0]
#             node.state = beliefState

#             # b. simulating
#             self.simulate(node)

#             it += 1

#         return node.get_best_action(self.target)

#     def planning(self, state, agent):  #  在当前的状态下选择一个最优动作输出
#         # 1. Getting the current state and previous action-observation pair
#         previous_action = agent.next_action
#         current_observation = state.get_observation()        

#         # 2. Defining the root of our search tree
#         if self.root is None:
#             self.root = ONode(observation=None,state=state,depth=0,parent=None)
#         # 来自真实环境的交互节点信息作为历史信息： 真实发生的线+线末端节点对应的树统计 信息
#         else:
#             self.root = find_new_PO_root(state, previous_action,\
#              current_observation, agent, self.root, adversary=self.adversary)

#         # 3. Performing particle revigoration
#         if self.pr:
#             particle_revigoration(state,agent,self.root,self.k)

#         # 4. Key: 从根节点开始多次 sim， 然后使用UCT 选择最优动作
#         best_action = self.search(self.root, agent)
#         # 6. Returning the best action
#         self.root.show_qtable()
#         info = { 'nrollouts': self.rollout_count,
#             'nsimulations':self.simulation_count}
#         return best_action, info



# def pomcp_planning(env, agent, max_depth=20, max_it=250, **kwargs):    
#     # 1. Setting the environment for simulation
#     copy_env = env.copy()
#     copy_env.simulation = True

#     # 2. POMCP Planning
#     pomcp = POMCP(max_depth, max_it, kwargs)

#     t0 = time.time()
#     next_action, info = pomcp.planning(copy_env, agent)
#     info = dict(info or {})
#     info["time_s"] = float(time.time() - t0)

#     # 3. Updating the search tree
#     agent.smart_parameters['pomcp'] = pomcp
#     agent.smart_parameters['count'] = info
#     return next_action,None
