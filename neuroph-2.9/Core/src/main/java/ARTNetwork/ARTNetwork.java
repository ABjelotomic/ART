/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ARTNetwork;



import java.util.ArrayList;
import java.util.List;
import org.neuroph.core.Connection;
import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.Weight;
import org.neuroph.core.transfer.Linear;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.ConnectionFactory;
import org.neuroph.util.LayerFactory;
import org.neuroph.util.NeuralNetworkFactory;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;
import org.neuroph.util.random.NguyenWidrowRandomizer;
import org.neuroph.core.input.WeightedSum;
import org.neuroph.core.transfer.Sigmoid;
import org.neuroph.nnet.comp.neuron.CompetitiveNeuron;
/**
 *
 * @author ja
 */
public class ARTNetwork extends NeuralNetwork <BackPropagation> {
    
    private double vigilance;
    private int L;
    public ARTNetwork (List<Integer> neuronsInLayers, double vigilance, int L ) {
        
		NeuronProperties neuronProperties = new NeuronProperties();
                neuronProperties.setProperty("useBias", true);
		neuronProperties.setProperty("transferFunction", TransferFunctionType.SIGMOID);
                neuronProperties.setProperty("inputFunction", new WeightedSum()); 

		this.createNetwork(neuronsInLayers, neuronProperties, vigilance, L);
    }
    
    
    public ARTNetwork (double vigilance, int L, int ... neuronsInLayers) {
		// init neuron settings
        
               
		NeuronProperties neuronProperties = new NeuronProperties();
                neuronProperties.setProperty("useBias", true);
		neuronProperties.setProperty("transferFunction",
				TransferFunctionType.SIGMOID);
                neuronProperties.setProperty("inputFunction", WeightedSum.class);

		List<Integer> neuronsInLayersVector = new ArrayList<>();
		for(int i=0; i<neuronsInLayers.length; i++) {
                    neuronsInLayersVector.add(new Integer(neuronsInLayers[i]));
                }
		
		this.createNetwork(neuronsInLayersVector, neuronProperties, vigilance, L);
	}
    
    
    public ARTNetwork (List<Integer> neuronsInLayers,NeuronProperties neuronProperties, double vigilance, int L) {
		this.createNetwork(neuronsInLayers, neuronProperties, vigilance, L);
	}
     
    
    
    private void createNetwork(List<Integer> neuronsInLayers, NeuronProperties neuronProperties, double vigilance, int L) {

                   
                this.vigilance = vigilance;
                
                NeuronProperties inputNeuronProperties = new NeuronProperties(InputNeuron.class, Linear.class);
                Layer f1a = LayerFactory.createLayer(neuronsInLayers.get(0), inputNeuronProperties);
                Layer f1b = LayerFactory.createLayer(neuronsInLayers.get(0), inputNeuronProperties);
                Layer f2 = LayerFactory.createLayer(neuronsInLayers.get(1), inputNeuronProperties);
                
                
                boolean useBias = true; 
                if (neuronProperties.hasProperty("useBias")) {
                    useBias = (Boolean)neuronProperties.getProperty("useBias");
                }  //da li mi je bitno i da li da zadrzim
                
                 
                
                if (useBias) {
                    f1a.addNeuron(new BiasNeuron());
                    f1b.addNeuron(new BiasNeuron());
                    f2.addNeuron(new CompetitiveNeuron(new WeightedSum(), new Sigmoid()));
                }

                this.addLayer(f1a);
                this.addLayer(f1b);
                this.addLayer(f2);
                
                
                ConnectionFactory.forwardConnect(f1a, f1b); //vidi je l to
                ConnectionFactory.fullConnect(f1b, f2);
                
                
                
                Neuron[] f1bneurons = f1b.getNeurons();
                
                for (int i = 0; i < f1bneurons.length; i++) {
                Neuron f1bneuron = f1bneurons[i];
                Connection[] f1bneuronConnections = f1bneuron.getOutConnections();
            
                    for (int j = 0; j < f1bneuronConnections.length; j++) {
                        Connection f1bneuronConnection = f1bneuronConnections[j];
                        Weight w = new Weight(L/(L-1 + neuronsInLayers.get(0)));
                        f1bneuronConnection.setWeight(w);
                        
                    }
            
        }
        
                Neuron[] f2neurons = f2.getNeurons();
                
                for (int i = 0; i < f2neurons.length; i++) {
                 Neuron f2neuron = f2neurons[i];
            
                Connection[] f2neuronConnections = f2neuron.getOutConnections();
            
                    for (int j = 0; j < f2neuronConnections.length; j++) {
                        Connection f2neuronConnection = f2neuronConnections[j];
                        Weight w = new Weight(1);
                        f2neuronConnection.setWeight(w);
                    }

            
        }
                        
                        

                  NeuralNetworkFactory.setDefaultIO(this);

                  
		this.setLearningRule(new MomentumBackpropagation());  //Koje da koristim
                
                
                this.randomizeWeights(new NguyenWidrowRandomizer(-0.7, 0.7));
				
	}
}
