package burlap.oomdp.singleagent.pomdp.beliefstate.stateparticles;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import burlap.behavior.singleagent.auxiliary.StateEnumerator;
import burlap.debugtools.DPrint;
import burlap.debugtools.RandomFactory;
import burlap.oomdp.core.Attribute;
import burlap.oomdp.core.ObjectClass;

import burlap.oomdp.core.objects.ObjectInstance;
import burlap.oomdp.core.states.MutableState;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.SADomain;
import burlap.oomdp.singleagent.pomdp.ObservationFunction;
import burlap.oomdp.singleagent.pomdp.PODomain;
import burlap.oomdp.singleagent.pomdp.beliefstate.BeliefState;
import burlap.oomdp.singleagent.pomdp.beliefstate.EnumerableBeliefState;
import burlap.oomdp.singleagent.pomdp.beliefstate.tabular.TabularBeliefState;
import burlap.oomdp.statehashing.HashableState;
import burlap.oomdp.statehashing.HashableStateFactory;


public class ParticleBeliefState implements BeliefState, EnumerableBeliefState{

	protected List<State> stateParticles = new ArrayList<State>();
	protected int maxNumberOfParticles = 16;
	protected int debugInt = 4325768;

	protected Random rand = RandomFactory.getMapped(1);

	protected boolean rejectionSamplingFailed = false;

	// in this class storing a belief vector might be impossible we are only storing the number of particles present
	private static final String BELIEFCLASSNAME = "belief";

	//TODO: right now this state has no attributes that represent the belief state at all, but given a common state enumerator we can store belief as a vector of enumerated states. Better for memory even?

	//	public static final String BELIEFTYPEATTNAME = "belief";

	private static final String NUMBEROFPARTICLESATTRIBUTENAME = "numberOfStateParticles";

	private static SADomain particleBeliefStateDomain = null;

	/**
	 * A state enumerator for determining the index of MDP states in the belief vector if given.
	 */
	protected StateEnumerator stateEnumerator = null;

	/**
	 * The POMDP domain with which this belief state is associated. Contains the {@link burlap.oomdp.singleagent.pomdp.ObservationFunction} necessary to perform
	 * belief state updates.
	 */
	protected PODomain domain;

	protected HashableStateFactory hsf;

	protected boolean stateEnumeratorPresent = false;

	public ParticleBeliefState(ParticleBeliefState srcBeliefParticleState){
		this.maxNumberOfParticles = srcBeliefParticleState.maxNumberOfParticles;
		this.domain = srcBeliefParticleState.domain;
		this.copyStateEnumerator(srcBeliefParticleState);
		this.hsf = srcBeliefParticleState.hsf;
		this.stateParticles = new ArrayList<State>(srcBeliefParticleState.stateParticles);		
	}


	public ParticleBeliefState(PODomain domainInput, StateEnumerator se, int numParticles, HashableStateFactory shf){
		this.maxNumberOfParticles = numParticles;
		this.domain = domainInput;
		this.stateEnumerator = se;
		this.hsf = shf;
	}

	public ParticleBeliefState(PODomain domainInput, HashableStateFactory shf, int numParticles){
		this.maxNumberOfParticles = numParticles;
		this.domain = domainInput;
		this.hsf = shf;
	}

	public ParticleBeliefState(PODomain domainInput, StateEnumerator se, HashableStateFactory shf, int numParticles){
		this.maxNumberOfParticles = numParticles;
		this.domain = domainInput;
		this.hsf = shf;
		this.stateEnumerator = se;
	}

	protected void copyStateEnumerator(ParticleBeliefState srcBeliefParticleState){
		if(srcBeliefParticleState.stateEnumeratorPresent){
			this.stateEnumerator = srcBeliefParticleState.stateEnumerator;
			this.stateEnumeratorPresent = true;
		}
	}


	public synchronized void addParticle(State s) {
		if(stateParticles.size()<maxNumberOfParticles){
			stateParticles.add(s);
		}
		else{
			throw new ArrayIndexOutOfBoundsException("Belief already has maximum number of particles allowed: " + stateParticles.size());
		}
	}

	public synchronized void removeRandomParticle() {
		stateParticles.remove(rand.nextInt(Integer.MAX_VALUE) % stateParticles.size());
	}

	public synchronized void removeIndexedParticle(int i) {
		if(i>this.maxNumberOfParticles){
			throw new ArrayIndexOutOfBoundsException("Index asked to be removed is greater than total number of particles.");
		}
		stateParticles.remove(i);
	}


	public  State sampleParticles() {
		return stateParticles.get(rand.nextInt(Integer.MAX_VALUE) % stateParticles.size());
	}

	public int particleCount() {
		return stateParticles.size();
	}

	public List<State> getParticles() {
		return this.stateParticles;
	}

	public void setParticles(List<State> particlesList) {
		if(particlesList.size()!=this.maxNumberOfParticles){
			System.err.println("The particle list being tried to set is not equal in size to the original particle set");
		}
		this.stateParticles = particlesList;
	}

	public synchronized void clearParticles() {
		stateParticles.clear();
	}

	@Override
	public State copy() {
		return new ParticleBeliefState(this);
	}

	@Override
	public State addObject(ObjectInstance o) {
		throw new UnsupportedOperationException("ParticleBeliefState cannot have OO-MDP objects added to it.");
	}

	@Override
	public State addAllObjects(Collection<ObjectInstance> objects) {
		throw new UnsupportedOperationException("ParticleBeliefState cannot have OO-MDP objects added to it.");
	}

	@Override
	public State removeObject(String oname) {
		throw new UnsupportedOperationException("ParticleBeliefState cannot have OO-MDP objects removed from it.");
	}

	@Override
	public State removeObject(ObjectInstance o) {
		throw new UnsupportedOperationException("ParticleBeliefState cannot have OO-MDP objects removed from it.");
	}

	@Override
	public State removeAllObjects(Collection<ObjectInstance> objects) {
		throw new UnsupportedOperationException("ParticleBeliefState cannot have OO-MDP objects removed from it.");
	}

	@Override
	public State renameObject(String originalName, String newName) {
		throw new UnsupportedOperationException("ParticleBeliefState cannot have OO-MDP objects renamed");
	}

	@Override
	public State renameObject(ObjectInstance o, String newName) {
		throw new UnsupportedOperationException("ParticleBeliefState cannot have OO-MDP objects renamed");
	}


	@Override
	public int numTotalObjects() {
		return 0;
	}

	@Override
	public ObjectInstance getObject(String oname) {
		throw new UnsupportedOperationException("ParticleBeliefState has only a list of states and no OOMDP objects.");
	}

	@Override
	public List<ObjectInstance> getAllObjects() {
		throw new UnsupportedOperationException("ParticleBeliefState has only a list of states and no OOMDP objects.");
	}

	@Override
	public List<ObjectInstance> getObjectsOfClass(String oclass) {
		throw new UnsupportedOperationException("ParticleBeliefState has only a list of states and no OOMDP objects.");
	}

	@Override
	public ObjectInstance getFirstObjectOfClass(String oclass) {
		throw new UnsupportedOperationException("ParticleBeliefState has only a list of states and no OOMDP objects.");
	}

	@Override
	public Set<String> getObjectClassesPresent() {
		throw new UnsupportedOperationException("ParticleBeliefState has only a list of states and no OOMDP objects.");
	}

	@Override
	public List<List<ObjectInstance>> getAllObjectsByClass() {
		throw new UnsupportedOperationException("ParticleBeliefState has only a list of states and no OOMDP objects.");
	}


	@Override
	public String getCompleteStateDescription() {
		throw new UnsupportedOperationException("ParticleBeliefState has only a list of states and no OOMDP objects.");
	}

	@Override
	public Map<String, List<String>> getAllUnsetAttributes() {
		return new HashMap<String, List<String>>();
	}

	@Override
	public String getCompleteStateDescriptionWithUnsetAttributesAsNull() {
		throw new UnsupportedOperationException("ParticleBeliefState has only a list of states and no OOMDP objects.");
	}

	@Override
	public List<List<String>> getPossibleBindingsGivenParamOrderGroups(
			String[] paramClasses, String[] paramOrderGroups) {
		System.out.println("ParticleBeliefState has no bindings or parameters.");
		return null;
	}

	@Override
	public List<StateBelief> getStatesAndBeliefsWithNonZeroProbability() {
		/** The original list of states */
		List<State> stateList = new ArrayList<State>(this.stateParticles);

		/** A list to count equalities of states so we don't recount for states we already checked for */
		List<Boolean> stateEqualityCount = new ArrayList<Boolean>(this.stateParticles.size());
		/** StateBelief list to be returned*/
		List<StateBelief> stateBeliefList = new ArrayList<StateBelief>();
		/** A list to count equal states so belief probabilities can be calculated */
		List<StateCount> stateCountList = new ArrayList<StateCount>();

		/** first none of the states are equal */
		for(int i = 0;i<stateEqualityCount.size();i++){
			stateEqualityCount.set(i, false);			
		}

		for(int i=0;i<stateList.size();i++){
			if(stateEqualityCount.get(i)){
				/** if state already counted before continue */
				continue;
			}
			double count = 1;
			for(int j=i+1;j<stateList.size();j++){
				/** Counting forward from the current state */
				if(this.hsf.hashState(stateList.get(i)).equals(this.hsf.hashState(stateList.get(j)))){
					/** if equal forward count and mark state as being equal to a previous one */
					count=count++;
					stateEqualityCount.set(j, true);
				}
			}

			stateCountList.add(new StateCount(stateList.get(i),count));

		}

		/** generating probabilities by counting */
		double totalCount = 0;
		for(StateCount sc:stateCountList){
			totalCount = totalCount+sc.count; 
		}

		for(StateCount sc: stateCountList){
			stateBeliefList.add(new StateBelief(sc.state,sc.count/totalCount));
		}

		return stateBeliefList;
	}

	@Override
	public double belief(State s) {
		HashableState stateQueried = this.hsf.hashState(s);
		double count = 0.0;
		for(State stateParticle:this.stateParticles){
			HashableState sTemp = this.hsf.hashState(stateParticle);
			if(stateQueried.equals(sTemp)){
				count+=1.0;
			}
		}
		return count/this.stateParticles.size();
	}

	@Override
	public State sampleStateFromBelief() {
		return this.sampleParticles();
	}

	@Override
	public BeliefState getUpdatedBeliefState(State obs, GroundedAction ga) {
		//TODO: set flag for failed update
		if(rejectionSamplingFailed){
//			ParticleBeliefState updatedBelief = new ParticleBeliefState(this.domain,this.stateEnumerator,this.hsf,this.maxNumberOfParticles);
			ParticleBeliefState updatedBelief = new ParticleBeliefState(this);
			updatedBelief.setRejectionSamplingFlag(rejectionSamplingFailed);
			return updatedBelief;
		}
		
		boolean observationPresentInTree = false;
		double totalProb = 0.;
		ObservationFunction of = domain.getObservationFunction(); 
		for(State obsTest : of.getAllPossibleObservations()){
			if(compareObservations(obsTest, obs)){
				observationPresentInTree = true;
				break;
			}
			
		}
		
		for(State s :this.stateParticles){
			totalProb += of.getObservationProbability(obs, s, ga);
		}
		
		if(totalProb==0.){
			observationPresentInTree  = false;
			DPrint.cl(debugInt, "-----------------------lost------------------------------");
		}
		
		
		

		//test something -> failed update and return an empty particle belief state

		ParticleBeliefState updatedBelief = new ParticleBeliefState(this.domain,this.stateEnumerator,this.hsf,this.maxNumberOfParticles);
		updatedBelief.setRejectionSamplingFlag(!observationPresentInTree);

		if(observationPresentInTree){
			while(updatedBelief.stateParticles.size()<updatedBelief.maxNumberOfParticles){
				State s = this.sampleParticles();
				State s_ = ga.executeIn(s);
				State o_ = ((PODomain)domain).getObservationFunction().sampleObservation(s_, ga);
				if(compareObservations(obs, o_)) updatedBelief.addParticle(s_);
			}
		}
		else{
			updatedBelief = new ParticleBeliefState(this);
			updatedBelief.setRejectionSamplingFlag(true);
		}
		return updatedBelief;
	}

	public void setRejectionSamplingFlag(boolean flagValue){
		rejectionSamplingFailed = flagValue;
	}

	private boolean compareObservations(State o1, State o2) {
		HashableState ho1 = hsf.hashState(o1);
		HashableState ho2 = hsf.hashState(o2);
		return ho1.equals(ho2);
	}

	public static class StateCount{
		public State state;
		public Double count;
		public StateCount(State s, double c){
			this.state=s;
			this.count = c;
		}
	}


	public PODomain getDomain(){
		return this.domain;
	}

	public StateEnumerator getStateEnumerator(){
		return this.stateEnumerator;
	}

	public HashableStateFactory getHashableStateFactory(){
		return this.hsf;
	}

	public Integer getMaximumParticlesSaved(){
		return this.maxNumberOfParticles;
	}

	public static SADomain getParticleBeliefMDPDomain(){
		if(particleBeliefStateDomain != null){
			return particleBeliefStateDomain;
		}
		particleBeliefStateDomain = new SADomain();
		Attribute att = new Attribute(particleBeliefStateDomain, NUMBEROFPARTICLESATTRIBUTENAME, Attribute.AttributeType.INT);
		ObjectClass oclass = new ObjectClass(particleBeliefStateDomain, BELIEFCLASSNAME);
		oclass.addAttribute(att);
		return particleBeliefStateDomain;
	}

	public TabularBeliefState getTabularBeliefState() {
		TabularBeliefState bs = new TabularBeliefState(this.domain); 

		StateEnumerator senum = (this.domain).getStateEnumerator();
		if(senum==null){
			System.err.println("ParticleBeliefState: getting belief state needs a state enumerator, which is not declared within the domain");
			return null;
		}
		double sumToAdd = 1.0/this.particleCount();

		// default belief points zero
		double[] beliefPoints = new double[senum.numStatesEnumerated()]; 
		for(State s : this.getParticles()){
			int temp = senum.getEnumeratedID(s);
			beliefPoints[temp] += sumToAdd; 
		}

		bs.setBeliefVector(beliefPoints);

		return bs;
	}


	public State getRandomState(){
		return stateParticles.get(rand.nextInt(maxNumberOfParticles));
	}



	@Override
	public <T> State setObjectsValue(
			String objectName, String attName, T value) {
		throw new UnsupportedOperationException("ParticleBeliefState cannot have OO-MDP objects set externally.");
	}




	@Override
	public Map<String, String> getObjectMatchingTo(State so, boolean enforceStateExactness) {
		throw new UnsupportedOperationException("ParticleBeliefState does not have OO-MDP objects, only a list of states.");
	}


	public boolean getRejectionSamplingStatus(){
		return rejectionSamplingFailed;
	}


}
