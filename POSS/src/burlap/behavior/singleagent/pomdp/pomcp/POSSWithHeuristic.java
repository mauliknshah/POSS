package burlap.behavior.singleagent.pomdp.pomcp;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;





import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.MDPSolver;
import burlap.debugtools.DPrint;
import burlap.debugtools.RandomFactory;
import burlap.domain.singleagent.pomdp.tiger.TigerDomain;
import burlap.oomdp.auxiliary.common.NullTermination;
import burlap.oomdp.core.AbstractGroundedAction;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.core.states.MutableState;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.Action;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.pomdp.SimulatedPOEnvironment;
import burlap.oomdp.singleagent.pomdp.beliefstate.BeliefState;
import burlap.oomdp.singleagent.pomdp.beliefstate.stateparticles.WeightedParticleBeliefState;
import burlap.oomdp.singleagent.pomdp.PODomain;
import burlap.oomdp.statehashing.HashableStateFactory;
import burlap.oomdp.statehashing.SimpleHashableStateFactory;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.pomdp.BeliefPolicyAgent;
import burlap.behavior.singleagent.pomdp.pomcp.HistoryElement;
import burlap.behavior.valuefunction.QFunction;
import burlap.behavior.valuefunction.QValue;



public class POSSWithHeuristic extends MDPSolver implements Planner, QFunction{


	protected int debugCode = 927490;

	protected WeightedMonteCarloNode root = null;



	private int NUM_PARTICLES = 16;
	private double EPSILON = 1E-2;
	private double EXP_BONUS = 5;
	private int NUM_SIMS = NUM_PARTICLES;
	private int BRANCHING = 8;
	private int HEIGHT = -1;


	private boolean randomPlannerSet = false;
	private List<Action> actionList; 

	private Random randomNumber = RandomFactory.getMapped(0);

	//observation Terminal Function
	private TerminalFunction  otf;

	private HeuristicForRandomRollouts heuristic;
	private boolean heuristicSet = false;

	/**
	 * 
	 * @param domainIn: input POMDP domain
	 * @param rfIn: reward function for the POMDP
	 * @param tfIn: Terminal Function for the MDP underneath
	 * @param otfIn: Terminal Function for the terminal observation
	 * @param discount: discount factor
	 * @param hashingFactory: hashing factory for the states
	 * @param explorationBonus: exploration constant C for the UCT to explore the tree underneath
	 * @param treeHeight: height of the tree to plan over. If set to -1 then runs to the depth of tree.
	 * @param branching: branching for the tree at the observation nodes.
	 */
	public POSSWithHeuristic(PODomain domainIn, RewardFunction rfIn, TerminalFunction tfIn, TerminalFunction otfIn, double discount, HashableStateFactory hashingFactory, double explorationBonus, int treeHeight,  int branching){
		super();
		this.EXP_BONUS=explorationBonus;
		this.BRANCHING = branching;
		this.domain = domainIn;
		this.rf = rfIn;
		this.tf = tfIn;
		this.gamma = discount;
		this.hashingFactory = hashingFactory;
		this.HEIGHT = treeHeight;
		this.otf = otfIn;
	}


	/**
	 * To use a heuristic set a heuristic and the length of action, observation length to be stored.
	 * @param hIn - input heuristic 
	 */
	public void setHeuristic(HeuristicForRandomRollouts hIn){
		this.heuristic = hIn;
		this.heuristicSet = true;
	}


	public List<QValue> getQs(State initialBeliefState) {
		if(!(initialBeliefState instanceof WeightedParticleBeliefState)){
			System.err.println("POSS needs a WeightedParticleBeliefState to plan from.");
			System.exit(-10);
		}
		if(!randomPlannerSet){
			planFromWeightedBeliefState((WeightedParticleBeliefState) initialBeliefState);
			return this.root.returnQVlauesForNode(initialBeliefState);
		}
		else{
			String str = "planning randomly";
			DPrint.cl(debugCode , str);
			DPrint.toggleCode(debugCode, false);
			List<GroundedAction> gaList = getGroundedActions(((WeightedParticleBeliefState)initialBeliefState).getRandomState());
			List<QValue> qList = new ArrayList<QValue>();
			for(GroundedAction ga: gaList){
				qList.add(new QValue(initialBeliefState, ga, 0.));
			}
			int gaChanged = randomNumber.nextInt(gaList.size());
			qList.set(gaChanged, new QValue(initialBeliefState, gaList.get(gaChanged), 0.));
			return qList;
		}
	}

	@Override
	public QValue getQ(State initialBeliefState, AbstractGroundedAction a) {
		if(!(initialBeliefState instanceof WeightedParticleBeliefState)){
			System.err.println("POSS needs a WeightedParticleBeliefState to plan from.");
			System.exit(-10);
		}
		if(!randomPlannerSet){
			planFromWeightedBeliefState((WeightedParticleBeliefState) initialBeliefState);
			return root.returnQVlaueForNode(initialBeliefState, a);
		}
		else{
			String str = "planning randomly";
			DPrint.cl(debugCode , str);
			DPrint.toggleCode(debugCode, false);
			return new QValue(initialBeliefState, a, 0.);
		}

	}


	public void planFromWeightedBeliefState(WeightedParticleBeliefState bs) {
		root = new WeightedMonteCarloNode((PODomain)domain, bs,hashingFactory);
		// create a WeightedMonteCarloNode with the same number of particles as needed.
		this.NUM_PARTICLES = bs.getMaximumParticlesSaved();
		this.NUM_SIMS = bs.getMaximumParticlesSaved();

		int simulations = 0;
		while(simulations < this.NUM_SIMS) {
			simulations++;
			State s = root.sampleParticles();

			if(s==null){
				this.randomPlannerSet = true;
				this.actionList = this.domain.getActions();
				break;

			}
			List<State> obsList = new ArrayList<State>();
			List<GroundedAction> gaList = new ArrayList<GroundedAction>();
			simulate(s, root, 0, null, null, obsList, gaList);
		}
	}


	private double simulate(State state, WeightedMonteCarloNode node, int depth, State _o, GroundedAction _ga, List<State> obsHistoryList, List<GroundedAction> gaHistoryList) {

		if(Math.pow(this.gamma, depth) < this.EPSILON ) return 0.;
		if(depth>this.HEIGHT && this.HEIGHT!=-1) return 0.;
		if(this.tf.isTerminal(state)) return 0.;
		if(_o!=null){
			if(this.otf.isTerminal(_o)) return 0;
			//			if(((PODomain)this.domain).getObservationFunction().isTerminalObservation(_o)) return 0;
		}

		if(node.isLeaf()) {
			List<GroundedAction> applicableGAList = getGroundedActions(state);
			if(applicableGAList.size() == 0) System.out.println("No actions for this state!");
			for(GroundedAction a : applicableGAList) {
				node.addChild(a);
			}

			double temp =  rollout(state, depth, obsHistoryList, gaHistoryList);
			return temp;
		}

		GroundedAction a = node.bestExploringAction(EXP_BONUS);
		gaHistoryList.add(0, a);
		State sPrime = (State) a.executeIn(state);
		State o = ((PODomain)this.domain).getObservationFunction().sampleObservation(sPrime, a);
		obsHistoryList.add(0, o);
		double r = this.rf.reward(state, a, sPrime);

		if(!node.advance(a).hasChild(o)){
			Map<HistoryElement,WeightedMonteCarloNode> children = node.advance(a).getMap();
			if(children.size()<=BRANCHING){
				node.advance(a).addChild(o);	
			}
			else{
				List<State> obsList = new ArrayList<State>();
				List<Double> obsProbabilityList = new ArrayList<Double>();
				List<Double> obsCDFList = new ArrayList<Double>();
				for(HistoryElement h : children.keySet()){
					State obsTemp = h.getObservation();
					obsList.add(obsTemp);
					obsProbabilityList.add(((PODomain)this.domain).getObservationFunction().getObservationProbability(obsTemp, sPrime, a));
				}
				listNorm(obsProbabilityList);
				obsCDFList = listCDF(obsProbabilityList);
				double tempRand = randomNumber.nextDouble();
				int ind=0;
				for(int count=0;count<obsCDFList.size();count++){
					if(tempRand < obsCDFList.get(count)){
						ind = count;
						break;
					}
				}
				o = null;
				o = obsList.get(ind);
			}

		}
		double expectedReward = r + this.gamma * simulate(sPrime, node.advance(a).advance(o), depth + 1, o , a, obsHistoryList, gaHistoryList);

		if(depth > 0 ) {
			node.addParticle(state,_o,_ga);
		}
		node.visit();
		node.advance(a).visit();
		node.advance(a).augmentValue((expectedReward - node.advance(a).getValue())/node.advance(a).getVisits());

		return expectedReward;

	}

	private double rollout(State state, int depth, List<State> obsHistoryList, List<GroundedAction> gaHistoryList) {
		if(Math.pow(this.gamma, depth) < this.EPSILON || this.tf.isTerminal(state)) {
			return 0;}
		if(depth>this.HEIGHT && this.HEIGHT!=-1) return 0.;

		State sPrime;
		GroundedAction a;
		if(this.heuristicSet==false){
			a = getGroundedActions(state).get(RandomFactory.getMapped(0).nextInt(getGroundedActions(state).size()));
			sPrime = (State) a.executeIn(state);
		}
		else{
			a = this.heuristic.MostLikelyAction(gaHistoryList, obsHistoryList, state);
			sPrime = (State) a.executeIn(state);
			State o = ((PODomain)this.domain).getObservationFunction().sampleObservation(sPrime, a);
			gaHistoryList.add(0, a);
			obsHistoryList.add(0, o);

		}



		double temp = this.rf.reward(state, a, sPrime) + this.gamma* rollout(sPrime, depth + 1,obsHistoryList, gaHistoryList);
		return temp;
	}

	private List<GroundedAction> getGroundedActions(State state) {
		List<GroundedAction> result = new ArrayList<GroundedAction>();
		for(Action a : domain.getActions()) {
			result.addAll(a.getAllApplicableGroundedActions(state));
		}
		return result;
	}

	private GroundedAction setRandomAction(State Obs){
		State realState = (MutableState)(Obs);
		List<GroundedAction> tempGaList = new ArrayList<GroundedAction>();
		for(Action aTemp : this.actionList){
			tempGaList.addAll(aTemp.getAllApplicableGroundedActions(realState));
		}
		int i = this.randomNumber.nextInt(tempGaList.size());

		return tempGaList.get(i);
	}


	@Override
	public Policy planFromState(State initialBeliefState) {
		//		if(!(initialBeliefState instanceof WeightedParticleBeliefState)){
		//			System.err.println("POSS needs a WeightedParticleBeliefState to plan from.");
		//			System.exit(-10);
		//		}
		//		planFromWeightedBeliefState((WeightedParticleBeliefState) initialBeliefState);

		return new GreedyQPolicy(this);
	}

	@Override
	public void resetSolver() {
		this.root.prune();
		this.root.clearBeliefState();
		this.root = null;
	}


	@Override
	public double value(State initialBeliefState) {
		if(!(initialBeliefState instanceof WeightedParticleBeliefState)){
			System.err.println("POSS needs a WeightedParticleBeliefState to plan from.");
			System.exit(-10);
		}
		if(!randomPlannerSet){
			planFromWeightedBeliefState((WeightedParticleBeliefState) initialBeliefState);
			return root.value;
		}
		String str = "planning randomly";
		DPrint.cl(debugCode , str);
		DPrint.toggleCode(debugCode, false);
		return 0.;


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

	public static class TigerHeuristic implements HeuristicForRandomRollouts{


		PODomain d;
		List<Action> aList;
		double noise = 0.;
		GroundedAction openLeft;
		GroundedAction openRight;
		GroundedAction listen;
		Random rand = new RandomFactory().getDefault();
		public TigerHeuristic(PODomain domain, double noise){
			d = domain;
			aList = d.getActions();
			this.noise = noise;
			for(Action a:aList){
				if(a.getName().equals(TigerDomain.ACTIONLEFT)){
					openLeft = a.getAssociatedGroundedAction();
				}
				if(a.getName().equals(TigerDomain.ACTIONRIGHT)){
					openRight = a.getAssociatedGroundedAction();
				}
				if(a.getName().equals(TigerDomain.ACTIONLISTEN)){
					listen = a.getAssociatedGroundedAction();
				}
			}
		}

		@Override
		public GroundedAction MostLikelyAction(List<GroundedAction> gaHistoryList, List<State> obsHistoryList, State currentState) {
			// TODO Auto-generated method stub
			// this is purely based on observations
			if(obsHistoryList.size()<2){
				return listen;
			}
			String obs0 = obsHistoryList.get(0).getFirstObjectOfClass(TigerDomain.CLASSOBSERVATION).getStringValForAttribute(TigerDomain.ATTOBSERVATION);
			String obs1 = obsHistoryList.get(1).getFirstObjectOfClass(TigerDomain.CLASSOBSERVATION).getStringValForAttribute(TigerDomain.ATTOBSERVATION);


			double tempRandom = rand.nextDouble();

			if(obs0.equals(TigerDomain.OBHEARLEFT)&& obs1.equals(TigerDomain.OBHEARLEFT)){
				if(tempRandom<noise){
					return listen;
				}
				else{
					return openLeft;
				}

			}
			else if(obs0.equals(TigerDomain.OBHEARLEFT)&& obs1.equals(TigerDomain.OBHEARRIGHT)){
				if(tempRandom<noise){
					if(rand.nextDouble()<0.5){
						return openLeft;
					}
					else{
						return openRight;
					}
				}
				else{
					return listen;
				}
			}
			else if(obs0.equals(TigerDomain.OBHEARRIGHT)&& obs1.equals(TigerDomain.OBHEARLEFT)){
				if(tempRandom<noise){
					if(rand.nextDouble()<0.5){
						return openLeft;
					}
					else{
						return openRight;
					}
				}
				else{
					return listen;
				}
			}
			else if(obs0.equals(TigerDomain.OBHEARRIGHT)&& obs1.equals(TigerDomain.OBHEARRIGHT)){
				if(tempRandom<noise){
					return listen;

				}
				else{
					return openRight;
				}
			}
			return listen;
		}

	}


	public static void main(String [] args){
		TigerDomain tiger = new TigerDomain(true);
		PODomain domain = (PODomain)tiger.generateDomain();
		BeliefState initialBelief = TigerDomain.getInitialBeliefState(domain);
		RewardFunction rf = new TigerDomain.TigerRF();
		TerminalFunction tf = new NullTermination();
		HashableStateFactory hsf = new SimpleHashableStateFactory();
		TerminalFunction otf = new NullTermination();
		WeightedParticleBeliefState wpbs = new WeightedParticleBeliefState(initialBelief, 512, domain, hsf);
		int treeHeight = 4;
		POSSWithHeuristic bss = new POSSWithHeuristic(domain, rf, tf, otf, 0.75, hsf, 1000, treeHeight,1);
		double noise = 0.;
		HeuristicForRandomRollouts heuristic = new TigerHeuristic(domain, noise);
		bss.setHeuristic(heuristic);
		Policy p = bss.planFromState(wpbs);

		SimulatedPOEnvironment env = new SimulatedPOEnvironment(domain, rf, tf);
		env.setCurStateTo(TigerDomain.tigerLeftState(domain));

		BeliefPolicyAgent agent = new BeliefPolicyAgent(domain, env, p);
		agent.setBeliefState(wpbs);



		agent.setEnvironment(env);

		/*
		State initialBeliefStateOb = BeliefMDPGenerator.getBeliefMDPState(bss.getBeliefMDP(), initialBelief);
		List<QValue> qs = bss.getQs(initialBeliefStateOb);
		for(QValue q : qs){
			System.out.println(q.a.toString() + ": " + q.q);
		}
		 */


		EpisodeAnalysis ea = agent.actUntilTerminalOrMaxSteps(30);

		for(int i = 0; i < ea.numTimeSteps()-1; i++){
			System.out.println(ea.getAction(i) + " " + ea.getReward(i+1));
		}


	}





}
