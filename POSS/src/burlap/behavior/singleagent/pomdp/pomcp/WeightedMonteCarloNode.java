package burlap.behavior.singleagent.pomdp.pomcp;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import burlap.behavior.valuefunction.QValue;
import burlap.debugtools.RandomFactory;
import burlap.oomdp.core.AbstractGroundedAction;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.pomdp.PODomain;
import burlap.oomdp.singleagent.pomdp.beliefstate.stateparticles.WeightedParticleBeliefState;
import burlap.oomdp.statehashing.HashableStateFactory;

public class WeightedMonteCarloNode {
	// this node needs a belief state, counts for value and visits
	// tree of children
	// needs a state hashing factory

	protected Map<HistoryElement, WeightedMonteCarloNode> children = new HashMap<HistoryElement, WeightedMonteCarloNode>();

	protected List<Double> valueHistory = new ArrayList<Double>();
	protected Random rand = RandomFactory.getMapped(0);
	
	protected HashableStateFactory shf;


	protected int visits;
	protected double value;
	protected WeightedParticleBeliefState belief;
	protected PODomain domain;
	
	public WeightedMonteCarloNode(PODomain d, WeightedParticleBeliefState pbs, HashableStateFactory SHF) {
		this.domain = d;
		this.belief = new WeightedParticleBeliefState(pbs);
		this.visits = 0;
		this.value = 0;
		this.shf = SHF;
	}


	public WeightedMonteCarloNode(PODomain d,int vis, double val, WeightedParticleBeliefState pbs, HashableStateFactory SHF) {
		this.domain = d;
		this.belief = new WeightedParticleBeliefState(pbs);
		this.visits = vis;
		this.value = val;
		this.shf = SHF;
	}
	

	public void pruneExcept(GroundedAction a) {
		pruneExcept(new HistoryElement(a,this.shf));
	}

	public void pruneExcept(State o) {
		pruneExcept(new HistoryElement(o,this.shf));
	}

	public void pruneExcept(HistoryElement h) {
		if(this.isLeaf()) return;

		List<HistoryElement> tbd = new ArrayList<HistoryElement>(); 
		for(HistoryElement elem : children.keySet()) {
			if(!elem.equals(h)) {
				children.get(elem).prune();
				tbd.add(elem);
			}
		}	
		for(HistoryElement elem : tbd) {
			children.remove(elem);
		}

	}

	public void prune() {
		if(this.isLeaf()) return;
		for(HistoryElement elem : children.keySet()) {
			children.get(elem).prune();
		}
		children.clear();
	}

	public synchronized void visit() {
		visits++;
	}

	public synchronized void augmentValue(double inc) {
		value += inc;
	}

	public synchronized void addParticle(State s, double b) {
		this.belief.addParticle(s, b);
	}
	
	public synchronized void addParticle(State s, State obs, GroundedAction ga) {
		this.belief.addParticle(s, obs, ga);
	}

	public synchronized void  saveValues() {
		valueHistory.add(value);
		for(HistoryElement he : children.keySet()) {
			children.get(he).saveValues();
		}
	}

	public synchronized  List<Double> getValueHistory() {
		return valueHistory;
	}


	public  State sampleParticles() {
		return this.belief.sampleParticles();
	}

	public int particleCount() {
		return this.belief.particleCount();
	}

	public GroundedAction bestRealAction() {
		if(this.isLeaf()) System.out.println("WeightedMonteCarloNode: Requested action from leaf... :(");

		double maxValue = Double.NEGATIVE_INFINITY;
		GroundedAction bestAction = null;

		for(HistoryElement h : children.keySet()) {
			if(children.get(h).getValue() > maxValue) {
				maxValue = children.get(h).getValue();
				bestAction = h.getAction();
			}
		}	

		return bestAction;
	}

	public synchronized GroundedAction bestExploringAction(double C) {
		double maxValue = Double.NEGATIVE_INFINITY;
		GroundedAction bestAction = null;

		for(HistoryElement h : children.keySet()) {
			WeightedMonteCarloNode child = children.get(h);
			int childVisitCount = child.getVisits();
			double test =Double.MAX_VALUE;
			if(childVisitCount > 0){

				test = child.getValue() + C * Math.sqrt(Math.log(this.getVisits()+1)/childVisitCount);
			}


			if(test > maxValue) {
				maxValue = test;
				bestAction = h.getAction();
			}
		}
		return bestAction;
	}
	
	public synchronized GroundedAction bestExploringAction(double C, int depth) {
		double maxValue = Double.NEGATIVE_INFINITY;
		GroundedAction bestAction = null;

		for(HistoryElement h : children.keySet()) {
			WeightedMonteCarloNode child = children.get(h);
			int childVisitCount = child.getVisits();
			double test =Double.MAX_VALUE;
			if(childVisitCount > 0){

				test = child.getValue() + C/(depth*depth*depth) * Math.sqrt(Math.log(this.getVisits()+1)/childVisitCount);
			}


			if(test > maxValue) {
				maxValue = test;
				bestAction = h.getAction();
			}
		}
		return bestAction;
	}

	public synchronized WeightedMonteCarloNode advance(GroundedAction a) {
		return advance(new HistoryElement(a,this.shf));
	}

	public synchronized WeightedMonteCarloNode advance(State o) {
		return advance(new HistoryElement(o,this.shf));
	}

	public synchronized WeightedMonteCarloNode advance(HistoryElement h) {
		return children.get(h);
	}

	public boolean hasChild(State o) {
		return children.containsKey(new HistoryElement(o,this.shf));
	}

	public boolean hasChild(GroundedAction a) {
		return children.containsKey(new HistoryElement(a,this.shf));
	}

	public synchronized void addChild(State o) {
		addChild(new HistoryElement(o,this.shf), 0, 0);
	}

	public synchronized void addChild(GroundedAction a) {
		addChild(new HistoryElement(a,this.shf), 0, 0);
	}

	public synchronized void addChild(HistoryElement h) {
		addChild(h, 0, 0);
	}

	public synchronized void addChild(State o, int vis, double val) {
		addChild(new HistoryElement(o,this.shf), vis, val);
	}

	public synchronized void addChild(GroundedAction a, int vis, double val) {
		addChild(new HistoryElement(a,this.shf), vis, val);
	}

	public synchronized void addChild(HistoryElement h, int vis, double val) {
		WeightedParticleBeliefState childBeliefStart = createEmptyChildParticleBeliefState(this.belief);
		this.children.put(h, new WeightedMonteCarloNode(this.domain ,vis, val,childBeliefStart,this.shf));
	}

	private WeightedParticleBeliefState createEmptyChildParticleBeliefState(WeightedParticleBeliefState parentBelief) {
		return new WeightedParticleBeliefState(parentBelief.getDomain(),parentBelief.getStateEnumerator(),this.getStateHashFactory(),parentBelief.getMaximumParticlesSaved());
	}

	private HashableStateFactory getStateHashFactory() {
		return this.shf;
	}


	public boolean isLeaf() {
		return this.children.isEmpty();
	}

	public int getVisits() {
		return this.visits;
	}

	public double getValue() {
		return this.value;
	}

	public List<State> getParticles() {
		return this.belief.getParticles();
	}

	public void setParticles(List<State> particlesList) {
		this.belief.setParticles(particlesList);
	}

	public Map<HistoryElement, WeightedMonteCarloNode> getMap() {
		return children;
	}

	public List<QValue> returnQVlauesForNode(State s){
		List<QValue> returnQValueList = new ArrayList<QValue>();
		for(HistoryElement h : this.children.keySet()){
			GroundedAction a = h.getAction();
			if(a==null){
				System.out.println("WeightedMonteCarloNode: Queried actions from a node without actions as children");
				return null;
			}
			else{
				returnQValueList.add(new QValue(s, a, this.advance(a).value));
			}
		}
		return returnQValueList;
	}
	
	public QValue returnQVlaueForNode(State s, AbstractGroundedAction a){
		GroundedAction ga = (GroundedAction)a;
		for(HistoryElement h : this.children.keySet()){
			if(h.getAction()==null){
				System.out.println("WeightedMonteCarloNode: Queried actions from a node without actions as children");
				return null;
			}
			
			
			else{
				if(ga.equals(h.getAction())){
					return new QValue(s, a, this.advance(h.getAction()).value);
				}
				
			}
		}
		return new QValue(s, a, 0.);
	}
	
	


	public void clearBeliefState() {
		this.prune();
		this.children.clear();
		this.belief.clearParticles();
	}
	
	public WeightedParticleBeliefState getParticleBeliefState() {
		return this.belief;
	}

}
