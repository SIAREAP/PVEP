from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

from .qlearn import create_etable, create_qtable, entropy, iucb_select_action, ucb_select_action


class Node:
    def __init__(self, state: Any, depth: int, parent: Optional["Node"] = None):
        self.state = state
        self.depth = int(depth)
        self.parent = parent
        self.children: List[Any] = []
        self.visits = 0

    def update_depth(self, new_depth: int) -> None:
        """
        Update the `depth` field for this subtree.

        Depth semantics (nl2pomdp convention):
        - O-nodes (observation/history nodes) count *environment steps / actions* from the root.
        - A-nodes (action nodes) are intermediate bookkeeping nodes and do NOT advance time.

        Therefore:
        - O -> A keeps the same depth
        - A -> O increments depth by 1
        """
        self.depth = int(new_depth)
        for c in self.children:
            try:
                # Action nodes do not advance the environment step counter.
                if isinstance(c, (ANode, IANode)):
                    c.update_depth(new_depth)
                else:
                    c.update_depth(new_depth + 1)
            except Exception:
                pass


class QNode(Node):
    def __init__(self, action: Any, state: Any, depth: int, parent: Optional[Node] = None):
        super().__init__(state, depth, parent)
        self.value = 0.0
        self.action = action
        self.actions = list(state.get_actions_list())
        self.qtable = create_qtable(self.actions)

    def update(self, action: Any, result: float) -> None:
        self.qtable[str(action)]["trials"] = int(self.qtable[str(action)]["trials"]) + 1
        self.qtable[str(action)]["sumvalue"] = float(self.qtable[str(action)]["sumvalue"]) + float(result)
        trials = float(self.qtable[str(action)]["trials"])
        qvalue = float(self.qtable[str(action)]["qvalue"])
        self.qtable[str(action)]["qvalue"] = qvalue + (float(result) - qvalue) / trials
        self.value += (float(result) - float(self.value)) / float(self.visits)

    def select_action(self, coef: float = 0.5, mode: str = "ucb") -> Any:
        if mode in ("ucb", "max", "ucb-max"):
            return ucb_select_action(self, c=coef, mode="max")
        if mode in ("min", "ucb-min"):
            return ucb_select_action(self, c=coef, mode="min")
        if mode in ("iucb", "iucb-max"):
            return iucb_select_action(self, alpha=coef, mode="max")
        if mode == "iucb-min":
            return iucb_select_action(self, alpha=coef, mode="min")
        raise NotImplementedError(f"Invalid best action mode: {mode}")

    def get_actions_prob_distribution(self, mode='max', max_reward=1):
        prob_distribution = {}
        
        norm = 0.0
        for a in self.qtable:
            if mode == 'max':
                prob_distribution[a] = self.qtable[a]['qvalue']
            elif mode == 'min':
                prob_distribution[a] = (max_reward-self.qtable[a]['qvalue'])
            else:
                raise NotImplemented
            norm += prob_distribution[a]
        
        if norm == 0.0:
            for a in prob_distribution:
                prob_distribution[a] = 1/len(prob_distribution)
        else:
            for a in prob_distribution:
                prob_distribution[a] /= norm
        return prob_distribution

    def get_best_action(self,mode='max'):
        # 1. Intialising the support variables
        # - maximisation
        if mode == 'max' or mode == 'ucb':
            target = 'max'
            best_action, bestQ = None, -100000000000
        # - minimisation
        elif mode == 'min' or mode == 'ucb-min':
            target = 'min'
            best_action, bestQ = None, 100000000000
        # - not implemented
        else:
            print('Invalid best action mode:',mode)
            raise NotImplemented

        # 2. Looking for the best action (max qvalue action)
        for a in self.actions:
            if target == 'max' and  \
             self.qtable[str(a)]['qvalue'] > bestQ  and \
             self.qtable[str(a)]['trials'] > 0:
                bestQ = self.qtable[str(a)]['qvalue']
                best_action = a
            elif target == 'min' and \
             self.qtable[str(a)]['qvalue'] < bestQ and \
             self.qtable[str(a)]['trials'] > 0:
                bestQ = self.qtable[str(a)]['qvalue']
                best_action = a

        # 3. Checking if a tie case exists
        tieCases = []
        for a in self.actions:
            if self.qtable[str(a)]['qvalue'] == bestQ:
                tieCases.append(a)

        if len(tieCases) > 1:
            # trying tie break by number of visits
            trials = [self.qtable[str(a)]['trials'] for a in tieCases]
            max_trial = max(trials)
            trialTieCases = []
            for a in tieCases:
                if self.qtable[str(a)]['trials'] == max_trial:
                    trialTieCases.append(a)

            if len(trialTieCases) > 1:
                best_action = random.choice(trialTieCases)
            else:
                best_action = trialTieCases[0]

        # 4. Returning the best action
        if best_action is None:
            best_action = random.sample(self.actions,1)[0]
        
        return best_action

    def show_qtable(self):
        print('%8s %8s %8s %8s' % ('Action','Q-Value','SumValue','Trials'))
        action_dict = {}
        for a in self.actions:
            action_dict[a] = [self.qtable[str(a)]['qvalue'],self.qtable[str(a)]['trials']]
        action_dict = sorted(action_dict,key=lambda x:(action_dict[x][0],action_dict[x][1]), reverse=True)
        
        for a in action_dict:
            if hasattr(self.state, 'action_dict'):
                print('%8s %8.4f %8.4f %8d' % (self.state.action_dict[a],self.qtable[str(a)]['qvalue'],\
                                            self.qtable[str(a)]['sumvalue'],self.qtable[str(a)]['trials']))
            else:
                print('%8s %8.4f %8.4f %8d' % (a,self.qtable[str(a)]['qvalue'],\
                                            self.qtable[str(a)]['sumvalue'],self.qtable[str(a)]['trials']))
        print('-----------------')
        print('%8s %8.4f %8s %8d' % ('Value',self.value,'Visits',self.visits) )
        print('-----------------')

class ANode(QNode):
    def __init__(self, action: Any, state: Any, depth: int, parent: Optional[Node] = None):
        super().__init__(action, state, depth, parent)
        self.action = action
        self.observation = None

    def add_child(self, observation: Any) -> "ONode":
        state = self.state.copy()
        child = ONode(observation, state, self.depth + 1, self)
        self.children.append(child)
        return child


class ONode(QNode):
    def __init__(self, observation: Any, state: Any, depth: int, parent: Optional[Node] = None):
        super().__init__(None, state, depth, parent)
        self.action = None
        self.observation = observation

        self.particle_filter: List[Any] = []
        self.particles_set: Dict[Any, Any] = {}


class IANode(ANode):
    def add_child(self, observation: Any) -> "IONode":
        state = self.state.copy()
        child = IONode(observation, state, self.depth + 1, self)
        self.children.append(child)
        return child


class IONode(ONode):
    def __init__(self, observation: Any, state: Any, depth: int, parent: Optional[Node] = None):
        # NOTE: In the original implementation, IONode calls ONode.__init__ with action=None.
        super().__init__(None, state, depth, parent)
        self.observation = observation

        self.particle_filter = []

        self.observation_distribution: Dict[Any, int] = {}
        self.entropy = 0.0
        self.cumentropy = 0.0
        self.max_entropy = 1.0

        self.etable = create_etable(self.actions)

    def add_child(self, state: Any, action: Any) -> IANode:
        # IONode -> IANode does NOT advance time (action edge).
        child = IANode(action, state, self.depth, self)
        self.children.append(child)
        return child

    def add_to_observation_distribution(self, observations_state: List[Any]) -> None:
        for state in observations_state:
            try:
                key = state.hash_observation()
            except Exception:
                key = repr(getattr(state, "get_observation", lambda: state)())
            self.observation_distribution[key] = int(self.observation_distribution.get(key, 0)) + 1

    def get_alpha(self) -> float:
        adjust_value = 0.2
        if int(self.visits) == 0:
            return 1.0 - adjust_value

        decaying_factor = math.e * math.log(float(self.visits)) / float(self.visits)
        entropy_value_trend = float(self.cumentropy) / (float(self.visits) * float(self.max_entropy))
        norm_entropy = decaying_factor * entropy_value_trend
        return (1.0 - 2.0 * adjust_value) * float(norm_entropy) + adjust_value

    def update(self, action: Any, result: float) -> None:
        # Q-table update
        self.qtable[str(action)]["trials"] = int(self.qtable[str(action)]["trials"]) + 1
        self.qtable[str(action)]["sumvalue"] = float(self.qtable[str(action)]["sumvalue"]) + float(result)
        trials = float(self.qtable[str(action)]["trials"])
        qvalue = float(self.qtable[str(action)]["qvalue"])
        self.qtable[str(action)]["qvalue"] = qvalue + (float(result) - qvalue) / trials
        self.value += (float(result) - float(self.value)) / float(self.visits)

        # Observation-table update
        result_entropy = float(entropy(self.observation_distribution))
        self.etable[str(action)]["trials"] = int(self.etable[str(action)]["trials"]) + 1
        self.etable[str(action)]["cumentropy"] = float(self.etable[str(action)]["cumentropy"]) + result_entropy
        self.cumentropy += result_entropy

        et_trials = float(self.etable[str(action)]["trials"])
        et_entropy = float(self.etable[str(action)]["entropy"])
        self.etable[str(action)]["entropy"] = et_entropy + (result_entropy - et_entropy) / et_trials
        self.entropy += (result_entropy - float(self.entropy)) / float(self.visits)

        if result_entropy > float(self.etable[str(action)]["max_entropy"]):
            self.etable[str(action)]["max_entropy"] = result_entropy

        if result_entropy > float(self.max_entropy):
            self.max_entropy = float(self.entropy)

    def get_best_action(self, alpha: float, mode: str = "iucb-max") -> Any:
        if mode in ("iucb-max", "iucb"):
            target = "max"
            best_action, best_q = None, -1e30
        elif mode == "iucb-min":
            target = "min"
            best_action, best_q = None, 1e30
        else:
            raise NotImplementedError(f"Invalid best action mode: {mode}")

        actions_q: Dict[str, float] = {}
        for a in self.actions:
            denom = float(self.etable[str(a)]["max_entropy"])
            denom = denom if denom > 0.0 else 1.0
            actions_q[str(a)] = (1.0 - float(alpha)) * float(self.qtable[str(a)]["qvalue"]) + (
                float(alpha) * float(self.etable[str(a)]["entropy"]) / denom
            )

            if target == "max" and actions_q[str(a)] > best_q and int(self.qtable[str(a)]["trials"]) > 0:
                best_q = actions_q[str(a)]
                best_action = a
            elif target == "min" and actions_q[str(a)] < best_q and int(self.qtable[str(a)]["trials"]) > 0:
                best_q = actions_q[str(a)]
                best_action = a

        tie_cases = [a for a in self.actions if actions_q.get(str(a), None) == best_q]
        if len(tie_cases) > 1:
            trials = [int(self.qtable[str(a)]["trials"]) for a in tie_cases]
            max_trial = max(trials)
            trial_tie_cases = [a for a in tie_cases if int(self.qtable[str(a)]["trials"]) == max_trial]
            best_action = random.choice(trial_tie_cases) if len(trial_tie_cases) > 1 else trial_tie_cases[0]
        elif len(tie_cases) == 1:
            best_action = tie_cases[0]

        if best_action is None:
            best_action = random.sample(self.actions, 1)[0]
        return best_action

    def show_qtable(self):
        print('%8s %8s %8s %8s | %8s %8s %8s' % \
            ('Action','Q-Value','SumValue','Trials','Entropy','MaxEntropy','NormEntropy'))
        action_dict = {}
        for a in self.actions:
            action_dict[a] = [self.qtable[str(a)]['qvalue'],\
                self.etable[str(a)]['entropy']/self.etable[str(a)]['max_entropy'],\
                    self.qtable[str(a)]['trials']]
        action_dict = sorted(action_dict,key=lambda x:(action_dict[x][0],action_dict[x][1],action_dict[x][2]), reverse=True)
        
        for a in action_dict:
            if hasattr(self.state, 'action_dict'):
                print('%8s %8.4f %8.4f %8d | %8.4f %8.4f %8.4f' \
                    % (self.state.action_dict[a],self.qtable[str(a)]['qvalue'],\
                    self.qtable[str(a)]['sumvalue'],self.qtable[str(a)]['trials'],\
                    self.etable[str(a)]['entropy'],self.etable[str(a)]['max_entropy'],\
                    self.etable[str(a)]['entropy']/self.etable[str(a)]['max_entropy']))
            else:
                print('%8s %8.4f %8.4f %8d' % (a,self.qtable[str(a)]['qvalue'],\
                                            self.qtable[str(a)]['sumvalue'],self.qtable[str(a)]['trials']))
        print('-----------------')
        print('%8s %8.4f %8s %8d' % ('Value',self.value,'Visits',self.visits) )
        print('-----------------')




def find_new_PO_root(
    current_state: Any,
    previous_action: Any,
    current_observation: Any,
    agent: Any,
    previous_root: Any,
    adversary: bool = False,
):
    """
    Tree advancement for (P)OMCP:
    Given the previous root and the *real* (action, observation) that just occurred,
    move the root to the matching child if it exists; otherwise restart a new root.

    Note: `adversary` is kept for API compatibility but is not used here.
    """
    _ = (agent, adversary)

    if previous_root is None:
        new_root = ONode(observation=None, state=current_state, depth=0, parent=None)
        return new_root

    action_node = None
    for child in getattr(previous_root, "children", []):
        if getattr(child, "action", None) == previous_action:
            action_node = child
            break
    if action_node is None:
        return ONode(observation=None, state=current_state, depth=0, parent=None)

    observation_node = None
    for child in getattr(action_node, "children", []):
        try:
            ok = child.state.observation_is_equal(current_observation)
        except Exception:
            ok = getattr(child, "observation", None) == current_observation
        if ok:
            observation_node = child
            break

    if observation_node is None:
        return ONode(observation=None, state=current_state, depth=0, parent=None)

    new_root = observation_node
    new_root.parent = None
    new_root.update_depth(0)
    return new_root

def particle_revigoration(env,agent,root,k):
    # 1. Copying the current root particle filter
    current_particle_filter = []
    for particle in root.particle_filter:
        current_particle_filter.append(particle)
    
    # 2. Reinvigorating particles for the new particle filter or
    # picking particles from the uniform distribution
    root.particle_filter = []
    if len(current_particle_filter) > 0: # particle ~ F_r
        while(len(root.particle_filter) < k):
            particle = random.sample(current_particle_filter,1)[0]
            root.particle_filter.append(particle)
    else: # particle ~ U
        while(len(root.particle_filter) < k):
            particle = env.sample_state(agent)
            root.particle_filter.append(particle)




__all__ = [
    "IANode",
    "IONode",
]







