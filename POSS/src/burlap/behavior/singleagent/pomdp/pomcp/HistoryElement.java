package burlap.behavior.singleagent.pomdp.pomcp;

import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.statehashing.HashableStateFactory;

/**
 * A history element is either an action or an observation branch as mentioned in the David Silver paper
 * @author ngopalan@cs.brown.edu
 *
 */

public class HistoryElement {
	private State observation = null;
	private GroundedAction action = null;
	private HashableStateFactory shf;

	public HistoryElement(State o, HashableStateFactory SHF) {
		observation = o;
		shf = SHF;
	}

	public HistoryElement(GroundedAction a, HashableStateFactory SHF) {
		action = a;
		shf = SHF;
	}

	public State getObservation() {
		return observation;
	}

	public GroundedAction getAction() {
		return action;
	}
	
	public HashableStateFactory getHashableStateFactory() {
		return shf;
	}

	public String getName() {
		if(observation != null) {
			return observation.getCompleteStateDescription();
		} else if(action != null) {
			String name = action.action.getName();
			return name;
		}
		return "";
	}

	@Override 
	public int hashCode() {
		if(action != null) {
			return action.hashCode();
		} 

		if(observation != null) {
			return shf.hashState(observation).hashCode();
		}

		return 0;
	}

	@Override
	public boolean equals(Object o) {
		if(!(o instanceof HistoryElement)) {
			return false;
		} else {
			HistoryElement h = (HistoryElement) o;
			if(h.getAction() != null) {
				return h.getAction().equals(this.action);
			}
			if(h.getObservation() != null) {
				return shf.hashState(h.getObservation()).equals(shf.hashState(this.getObservation()));
			}
		}
		return false;
	}
}

