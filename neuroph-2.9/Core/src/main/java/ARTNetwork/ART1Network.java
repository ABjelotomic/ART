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
public class ART1Network extends NeuralNetwork <ART1Learning> {
    
    
    
    /*  @param vigilance
    *           controls whether concrete or general categories are learned
    *
    */
    private double vigilance;
    
    private int L;
    
    
    /** Creates new ART1 Network, with specified number of neurons in layers
    *   and a vigilance parameter
    * 
    *  @param neuronsInLayers
    *            collection of neuron number in layers
    * 
    */
    
    public ART1Network (List<Integer> neuronsInLayers, double vigilance, int L ) {
        
		NeuronProperties neuronProperties = new NeuronProperties();
                neuronProperties.setProperty("useBias", true);
		neuronProperties.setProperty("transferFunction", TransferFunctionType.SIGMOID);
                neuronProperties.setProperty("inputFunction", new WeightedSum()); 

		this.createNetwork(neuronsInLayers, neuronProperties, vigilance, L);
    }
    
    
    public ART1Network (double vigilance, int L, int ... neuronsInLayers) {
		// init neuron settings
        
               
		NeuronProperties neuronProperties = new NeuronProperties();
                neuronProperties.setProperty("useBias", true);
		neuronProperties.setProperty("transferFunction",
				TransferFunctionType.SIGMOID);
                neuronProperties.setProperty("inputFunction", WeightedSum.class);
                
                // This makes a vector, which gives as an array of numbers of neurons in each layer
                
		List<Integer> neuronsInLayersVector = new ArrayList<>();
		for(int i=0; i<neuronsInLayers.length; i++) {
                    neuronsInLayersVector.add(new Integer(neuronsInLayers[i]));
                }
		
		this.createNetwork(neuronsInLayersVector, neuronProperties, vigilance, L);
	}
    
    
    public ART1Network (List<Integer> neuronsInLayers,NeuronProperties neuronProperties, double vigilance, int L) {
		this.createNetwork(neuronsInLayers, neuronProperties, vigilance, L);
	}
     
    /**
	 * Creates ART Network architecture
	 *                The first (input portion, F1a) and the second (interface portion, F1b) layers are
         *                connected with forward, one-directional connection  
         *                while, the second and the third, competitive (F2) layer 
         *                are fully connected, in both directions
         * 
	 * 
	 * @param neuronsInLayers
	 *            collection of neuron numbers in getLayersIterator
	 * @param neuronProperties
	 *            neuron properties
	 */
    
    private void createNetwork(List<Integer> neuronsInLayers, NeuronProperties neuronProperties, double vigilance, int L) {

                   
                this.vigilance = vigilance;
                
                /** create layers
                 *  Layer f1a and f1b have the same number of neurons 
                 *  
                 */ 
                NeuronProperties inputNeuronProperties = new NeuronProperties(InputNeuron.class, Linear.class);
                Layer f1a = LayerFactory.createLayer(neuronsInLayers.get(0), inputNeuronProperties);
                Layer f1b = LayerFactory.createLayer(neuronsInLayers.get(0), inputNeuronProperties);
                Layer f2 = new ARTF2Layer(neuronsInLayers.get(1), inputNeuronProperties);
                
                // use bias neuron by default
                boolean useBias = true; 
                if (neuronProperties.hasProperty("useBias")) {
                    useBias = (Boolean)neuronProperties.getProperty("useBias");
                }  
            
                if (useBias) {
                    f1a.addNeuron(new BiasNeuron());
                    f1b.addNeuron(new BiasNeuron());
                    f2.addNeuron(new CompetitiveNeuron(new WeightedSum(), new Sigmoid()));
                }
                
                // add layers to the network
                this.addLayer(f1a);
                this.addLayer(f1b);
                this.addLayer(f2);
                
                /** Connecting layers
                 *  F1a sends the input to F1b, via forward connection
                 *  Each neuron from F1b layer communicates with each
                 *  neuron in F2, and the other way around (bottom-up
                 * and top-down weights)
                 * 
                 * 
                 * Step 5.
                 * Sending input signal from F1a to F1b
                 */
                ConnectionFactory.forwardConnect(f1a, f1b);
                ConnectionFactory.fullConnect(f1b, f2);
                ConnectionFactory.fullConnect(f2, f1b);
                
                
                
                /** Step 0.
                 * Initializing weights
                 * 
                 * Making an array of F1b neurons
                */
                Neuron[] f1bneurons = f1b.getNeurons();
          
                
                
        // Initializing bottom-up weights        
        for (Neuron f1bneuron : f1bneurons) {
            Connection[] f1bneuronConnections = f1bneuron.getOutConnections();
            
            for (Connection f1bneuronConnection : f1bneuronConnections) {
                Weight w = new Weight(L/(L-1 + neuronsInLayers.get(0)));
                f1bneuronConnection.setWeight(w);
            }
        } //od f1a do f1b da li je po difoltu weight 1, ili mora da se prodje loopom
        //da bi se namestile 1
                    
                //Making an array of F2b neurons
                Neuron[] f2neurons = f2.getNeurons();
         
        // Initializing top-down weights        
        for (Neuron f2neuron : f2neurons) {
            Connection[] f2neuronConnections = f2neuron.getOutConnections();
            
            for (Connection f2neuronConnection : f2neuronConnections) {
                Weight w = new Weight(1);
                f2neuronConnection.setWeight(w);
            }
        }
                        
                        

                  NeuralNetworkFactory.setDefaultIO(this);

                  
		this.setLearningRule(new ART1Learning());  
                
                
				
	}

    public double getVigilance() {
        return vigilance;
    }

    public void setVigilance(double vigilance) {
        this.vigilance = vigilance;
    }

    public int getL() {
        return L;
    }

    public void setL(int L) {
        this.L = L;
    }
    
    
}
