 /*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ARTNetwork;

import org.neuroph.core.Connection;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.core.learning.UnsupervisedLearning;
import org.neuroph.nnet.comp.neuron.CompetitiveNeuron;




public class ART1Learning extends UnsupervisedLearning {

    

    
    
    public ART1Learning() {
		super();
	}

    
	

	/**
	 * This method does one learning epoch for the unsupervised learning rules.
	 * It iterates through the training set and trains network weights for each
	 * element. Stops learning after one epoch.
	 * 
	 * @param trainingSet
	 *            training set for training network
	 */
	@Override
	public void doLearningEpoch(DataSet trainingSet) {
		super.doLearningEpoch(trainingSet);
		stopLearning(); // stop learning ahter one learning epoch - because we dont have any stopping criteria  for unsupervised...
	}		
	
	/**
	 * Adjusts weights for the winning neuron
	 */
	protected void updateNetworkWeights() {
		// find active neuron in output layer
            
		CompetitiveNeuron winningNeuron = ((org.neuroph.nnet.comp.layer.CompetitiveLayer) neuralNetwork
				.getLayerAt(1)).getWinner();

		Connection[] inputConnections = winningNeuron
				.getConnectionsFromOtherLayers();

		for(Connection connection : inputConnections) {
			double weight = connection.getWeight().getValue();
			double input = connection.getInput();
			double deltaWeight = this.learningRate * (input - weight);
			connection.getWeight().inc(deltaWeight);			
		}
	}
    
}
