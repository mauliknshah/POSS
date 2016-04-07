package burlap.behavior.singleagent.pomdp.pomcp;

import java.util.List;

import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.GroundedAction;

/**
 * This is a heuristic to provide an action with maximum likelihood based on the actions performed and
 * observations seen during the rollout.
 * @author ngopalan
 *
 */


public interface HeuristicForRandomRollouts {
	
	/**
	 * This is a list of observations and actions taken previously in the rollout tree. Depending on 
	 * the previous action, observation distribution for the particle some actions are more 
	 * likely than others. This interface is written for particle based solvers like POSS and POMCP, 
	 * but other belief state based solvers can use the same. The action observation list have the same size.
	 * @param gaList - List of actions taken
	 * @param obsList - List of observations see during the rollout.
	 * @param currentState - This can be useful if rollouts are directed with respected to the particle sampled.
	 * @return ga -  return the most likely grounded action for the random rollout.
	 */
	GroundedAction MostLikelyAction(List<GroundedAction> gaList, List<State> obsList, State currentState);

}
