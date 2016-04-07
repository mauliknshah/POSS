package burlap.oomdp.singleagent.pomdp.beliefstate.stateparticles;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import burlap.behavior.singleagent.auxiliary.StateEnumerator;

import burlap.debugtools.RandomFactory;
import burlap.oomdp.core.Attribute;
import burlap.oomdp.core.ObjectClass;

import burlap.oomdp.core.objects.ObjectInstance;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.SADomain;
import burlap.oomdp.singleagent.pomdp.PODomain;
import burlap.oomdp.singleagent.pomdp.beliefstate.BeliefState;
import burlap.oomdp.singleagent.pomdp.beliefstate.EnumerableBeliefState;
import burlap.oomdp.singleagent.pomdp.beliefstate.tabular.TabularBeliefState;
import burlap.oomdp.statehashing.HashableState;
import burlap.oomdp.statehashing.HashableStateFactory;


public class WeightedParticleBeliefState implements BeliefState, EnumerableBeliefState{

	protected List<State> stateParticles = new ArrayList<State>();
	protected List<Double> particleWeights = new ArrayList<Double>();
	protected int maxNumberOfParticles = 16;
	private List<Double> weightCDF = new ArrayList<Double>();
	

	protected Random rand = RandomFactory.getMapped(1);
	
	private boolean listNormalizedFlag = false;

	// in this class storing a belief vector might be impossible we are only storing the number of particles present
	private static final String BELIEFCLASSNAME = "belief";
	
	//	public static final String BELIEFTYPEATTNAME = "belief";

	private static final String NUMBEROFPARTICLESATTRIBUTENAME = "numberOfStateParticles";

	private static SADomain weightedParticleBeliefStateDomain = null;

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
	
	public WeightedParticleBeliefState(WeightedParticleBeliefState srcBeliefParticleState){
		this.maxNumberOfParticles = srcBeliefParticleState.maxNumberOfParticles;
		this.domain = srcBeliefParticleState.domain;
		this.copyStateEnumerator(srcBeliefParticleState);
		this.hsf = srcBeliefParticleState.hsf;
		this.stateParticles = new ArrayList<State>(srcBeliefParticleState.stateParticles);		
		this.particleWeights = new ArrayList<Double>(srcBeliefParticleState.particleWeights);		
	}
	
	public WeightedParticleBeliefState(BeliefState srcBeliefState,  int numberOfParticles, PODomain inputDomain, HashableStateFactory inputHsf){
		this.maxNumberOfParticles = numberOfParticles;
		this.domain = inputDomain;
		this.hsf = inputHsf;
		this.stateParticles = new ArrayList<State>();		
		this.particleWeights = new ArrayList<Double>();		
		for(int i=0;i<maxNumberOfParticles;i++){
			this.stateParticles.add(srcBeliefState.sampleStateFromBelief());
			this.particleWeights.add(1.0/maxNumberOfParticles);
		}
	}
	
	public WeightedParticleBeliefState(BeliefState srcBeliefState,  int numberOfParticles, PODomain inputDomain, StateEnumerator se, HashableStateFactory inputHsf){
		this.maxNumberOfParticles = numberOfParticles;
		this.stateEnumerator = se;
		this.domain = inputDomain;
		this.hsf = inputHsf;
		this.stateParticles = new ArrayList<State>();		
		this.particleWeights = new ArrayList<Double>();		
		for(int i=0;i<maxNumberOfParticles;i++){
			this.stateParticles.add(srcBeliefState.sampleStateFromBelief());
			this.particleWeights.add(1.0/maxNumberOfParticles);
		}
	}

	public WeightedParticleBeliefState(PODomain domainInput, StateEnumerator se, int numParticles, HashableStateFactory shf){
		this.maxNumberOfParticles = numParticles;
		this.domain = domainInput;
		this.stateEnumerator = se;
		this.hsf = shf;
	}

	public WeightedParticleBeliefState(PODomain domainInput, HashableStateFactory shf, int numParticles){
		this.maxNumberOfParticles = numParticles;
		this.domain = domainInput;
		this.hsf = shf;
	}

	public WeightedParticleBeliefState(PODomain domainInput, StateEnumerator se, HashableStateFactory shf, int numParticles){
		this.maxNumberOfParticles = numParticles;
		this.domain = domainInput;
		this.hsf = shf;
		this.stateEnumerator = se;
	}

	protected void copyStateEnumerator(WeightedParticleBeliefState srcBeliefParticleState){
		if(srcBeliefParticleState.stateEnumeratorPresent){
			this.stateEnumerator = srcBeliefParticleState.stateEnumerator;
			this.stateEnumeratorPresent = true;
		}
	}


	public synchronized void addParticle(State s, double bs) {
		if(stateParticles.size()<maxNumberOfParticles){
			stateParticles.add(s);
			particleWeights.add(bs);
			listNormalizedFlag = false;
		}
		else{
			throw new ArrayIndexOutOfBoundsException("Belief already has maximum number of particles allowed.");
		}
	}
	
	public synchronized void addParticle(State s, State o, GroundedAction ga) {
		if(stateParticles.size()<maxNumberOfParticles){
			stateParticles.add(s);
			particleWeights.add(((PODomain)this.domain).getObservationFunction().getObservationProbability(o, s, ga));
			listNormalizedFlag = false;
		}
		else{
			throw new ArrayIndexOutOfBoundsException("Belief already has maximum number of particles allowed.");
		}
	}
	

	public  State sampleParticles() {
		if(!listNormalizedFlag){
			this.normalizeWeights();
		}
		double temp = rand.nextDouble();
		for (int count=0;count< weightCDF.size();count++){
//			System.out.println("CDF value: index- "+count + ", value: "+weightCDF.get(count));
			if (temp < weightCDF.get(count)) {
//				System.out.println("random number: " + temp + " index returned " + count);
				return stateParticles.get(count);
			}
			
		}
		System.err.println("WeightedParticleBeliefState: sampleparticles weights not summing to 1, tempRandom: " + temp + ", lastCountOfCDF: "+ weightCDF.get(weightCDF.size()-1));
//		for (int count=0;count<weightCDF.size();count++){
//			System.out.println(weightCDF.get(count) + ", ");
//		}
			
		return null;
	}

	public int particleCount() {
		return stateParticles.size();
	}

	public List<State> getParticles() {
		return this.stateParticles;
	}
	
	public List<Double> getParticleWeights() {
		return this.particleWeights;
	}

	public void setParticles(List<State> particlesList) {
		if(particlesList.size()!=this.maxNumberOfParticles){
			System.err.println("The particle list being tried to set is not equal in size to the original particle set");
		}
		this.stateParticles = particlesList;
	}
	
	public synchronized void clearParticles() {
		stateParticles.clear();
		particleWeights.clear();
		listNormalizedFlag =  false;
	}

	@Override
	public State copy() {
		return new WeightedParticleBeliefState(this);
	}

	@Override
	public State addObject(ObjectInstance o) {
		throw new UnsupportedOperationException("WeightedParticleBeliefState cannot have OO-MDP objects added to it.");
	}

	@Override
	public State addAllObjects(Collection<ObjectInstance> objects) {
		throw new UnsupportedOperationException("WeightedParticleBeliefState cannot have OO-MDP objects added to it.");
	}

	@Override
	public State removeObject(String oname) {
		throw new UnsupportedOperationException("WeightedParticleBeliefState cannot have OO-MDP objects removed from it.");
	}

	@Override
	public State removeObject(ObjectInstance o) {
		throw new UnsupportedOperationException("WeightedParticleBeliefState cannot have OO-MDP objects removed from it.");
	}

	@Override
	public State removeAllObjects(Collection<ObjectInstance> objects) {
		throw new UnsupportedOperationException("WeightedParticleBeliefState cannot have OO-MDP objects removed from it.");
	}

	@Override
	public State renameObject(String originalName, String newName) {
		throw new UnsupportedOperationException("WeightedParticleBeliefState cannot have OO-MDP objects renamed");
	}

	@Override
	public State renameObject(ObjectInstance o, String newName) {
		throw new UnsupportedOperationException("WeightedParticleBeliefState cannot have OO-MDP objects renamed");
	}


	@Override
	public int numTotalObjects() {
		return 0;
	}

	@Override
	public ObjectInstance getObject(String oname) {
		throw new UnsupportedOperationException("WeightedParticleBeliefState has only a list of states and no OOMDP objects.");
	}

	@Override
	public List<ObjectInstance> getAllObjects() {
		throw new UnsupportedOperationException("WeightedParticleBeliefState has only a list of states and no OOMDP objects.");
	}

	@Override
	public List<ObjectInstance> getObjectsOfClass(String oclass) {
		throw new UnsupportedOperationException("WeightedParticleBeliefState has only a list of states and no OOMDP objects.");
	}

	@Override
	public ObjectInstance getFirstObjectOfClass(String oclass) {
		throw new UnsupportedOperationException("WeightedParticleBeliefState has only a list of states and no OOMDP objects.");
	}

	@Override
	public Set<String> getObjectClassesPresent() {
		throw new UnsupportedOperationException("WeightedParticleBeliefState has only a list of states and no OOMDP objects.");
	}

	@Override
	public List<List<ObjectInstance>> getAllObjectsByClass() {
		throw new UnsupportedOperationException("WeightedParticleBeliefState has only a list of states and no OOMDP objects.");
	}


	@Override
	public String getCompleteStateDescription() {
		throw new UnsupportedOperationException("WeightedParticleBeliefState has only a list of states and no OOMDP objects.");
	}

	@Override
	public Map<String, List<String>> getAllUnsetAttributes() {
		return new HashMap<String, List<String>>();
	}

	@Override
	public String getCompleteStateDescriptionWithUnsetAttributesAsNull() {
		throw new UnsupportedOperationException("WeightedParticleBeliefState has only a list of states and no OOMDP objects.");
	}

	@Override
	public List<List<String>> getPossibleBindingsGivenParamOrderGroups(
			String[] paramClasses, String[] paramOrderGroups) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<StateBelief> getStatesAndBeliefsWithNonZeroProbability() {
		/** The original list of states */
		if(!this.listNormalizedFlag){
			this.normalizeWeights();
		}
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
			double count = this.particleWeights.get(i);
			for(int j=i+1;j<stateList.size();j++){
				/** Counting forward from the current state */
				if(this.hsf.hashState(stateList.get(i)).equals(this.hsf.hashState(stateList.get(j)))){
					/** if equal forward count and mark state as being equal to a previous one */
					count+=this.particleWeights.get(j);
					stateEqualityCount.set(j, true);
				}
			}

			stateBeliefList.add(new StateBelief(stateList.get(i),count));

		}
		
		return stateBeliefList;
	}

	@Override
	public double belief(State s) {
		if(!this.listNormalizedFlag){
			this.normalizeWeights();
		}
		
		HashableState stateQueried = this.hsf.hashState(s);
		double count = 0.0;
		for(int i =0;i<this.stateParticles.size();i++){
			HashableState sTemp = this.hsf.hashState(stateParticles.get(i));
			if(stateQueried.equals(sTemp)){
				count+=particleWeights.get(i);
			}
		}
		return count;
	}

	@Override
	public State sampleStateFromBelief() {
		return this.sampleParticles();
	}

	@Override
	public BeliefState getUpdatedBeliefState(State obs, GroundedAction ga) {
		WeightedParticleBeliefState updatedBelief = new WeightedParticleBeliefState(this.domain,this.stateEnumerator,this.hsf,this.maxNumberOfParticles);
		while(updatedBelief.stateParticles.size()<updatedBelief.maxNumberOfParticles){
			State s = this.sampleParticles();
			State s_ = ga.executeIn(s);
			updatedBelief.addParticle(s_, obs, ga);
		}
		return updatedBelief;
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
	
	public static SADomain getWeightedParticleBeliefMDPDomain(){
		if(weightedParticleBeliefStateDomain != null){
			return weightedParticleBeliefStateDomain;
		}
		weightedParticleBeliefStateDomain = new SADomain();
		Attribute att = new Attribute(weightedParticleBeliefStateDomain, NUMBEROFPARTICLESATTRIBUTENAME, Attribute.AttributeType.INT);
		ObjectClass oclass = new ObjectClass(weightedParticleBeliefStateDomain, BELIEFCLASSNAME);
		oclass.addAttribute(att);
		return weightedParticleBeliefStateDomain;
	}
	
	public TabularBeliefState getTabularBeliefState() {
		this.normalizeWeights();
		TabularBeliefState bs = new TabularBeliefState(this.domain); 

		StateEnumerator senum = (this.domain).getStateEnumerator();
		
		if(senum==null){
			System.err.println("WeightedMonteCarloNode: getting belief state needs a state enumerator, which is not declared within the domain");
			return null;
		}
		double[] beliefPoints = new double[senum.numStatesEnumerated()]; 
		for(int i = 0;i<this.stateParticles.size();i++){
			State s = this.stateParticles.get(i);
			int temp = senum.getEnumeratedID(s);
			beliefPoints[temp] += this.particleWeights.get(i); 
		}
		bs.setBeliefVector(beliefPoints);
		return bs;
	}


	
	@Override
	public <T> State setObjectsValue(
			String objectName, String attName, T value) {
		throw new UnsupportedOperationException("WeightedParticleBeliefState cannot have OO-MDP objects set externally.");
	}

	


	@Override
	public Map<String, String> getObjectMatchingTo(State so, boolean enforceStateExactness) {
		throw new UnsupportedOperationException("WeightedParticleBeliefState does not have OO-MDP objects, only a list of states.");
	}
	
	
	public static void listNorm(List<Double> list) {
		double sum = 0.0;
		for(int i = 0; i < list.size(); ++i) {
			sum += list.get(i);
		}
		if(sum==0.0){
			System.err.println("WeightedParticleBeliefState: sum of weights 0");
		}
		for(int i = 0; i < list.size(); ++i) {
			list.set(i, list.get(i)/sum);
		}
	}
	
	public static List<Double> listCDF(List<Double> list) {
		double sum = 0.0;
		List<Double> returnList = new ArrayList<Double>(); 
		for(int i = 0; i < list.size(); ++i) {
			sum += list.get(i);
			returnList.add(sum);
		}
		return returnList;
	}
	
	public void normalizeWeights(){
		if(this.particleWeights.size()==0){
			System.err.println("No weights added");
			System.exit(-1);
		}
		if(this.particleWeights.size()!=this.stateParticles.size()){
			System.err.println("Weights list does not have the same size as the belief particles list");
			System.exit(-1);
		}
		listNorm(this.particleWeights);
		this.weightCDF.addAll(listCDF(particleWeights));
		listNormalizedFlag = true;
	}
	
	public State getRandomState(){
		return stateParticles.get(rand.nextInt(maxNumberOfParticles));
	}



}
