package burlap.behavior.singleagent.pomdp.nbc;

import java.util.List;


import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.pomdp.beliefstate.BeliefState;



public interface NBCFeatureDatabase{
		
		/**
		 * Returns non-zero state features for a given state. This method should be implemented if it is to be used with algorithms
		 * that compute V-values from state features or other values that are independent of the actions.
		 * @param s the state for which features should be returned
		 * @return the features for state s
		 * 
		 * right now only returning the feature values no ids
		 */
		
		
		
		/**
		 * Returns non-zero action features for all of provided actions in state s. This method should be implemented if it is to be used 
		 * with algorithms that compute Q-values from action features or other values are the dependent on the actions.
		 * Note that features for different actions should return different feature ids.
		 * @param s the state for which features should be returned
		 * @param actions the action set for which the action features should be returned
		 * @return returns the set set of action features for each action in actions
		 * 
		 * right now only returning the feature values no ids
		 */
		public double[] getActionFeaturesSets(BeliefState bs, GroundedAction a);
		
		
		/**
		 * Enable or disable the database from generating new features. This method only needs to do anything if it creates features in an
		 * online fashion. That is, if the set of possible state features is defined at construction, then this method does not need to do anything.
		 * If features are generated as needed, such as with an instance-based feature set, then when the database is told to be frozen no new
		 * instances should be created.
		 * @param toggle
		 */
//		public void freezeDatabaseState(boolean toggle);
		
		public int getSizeOfBeliefFeatureVector();
		
		/**
		 * Returns the number of features this database tracks.
		 * @return the number of features this database tracks.
		 */
//		public int numberOfFeatures();
		
		
}


