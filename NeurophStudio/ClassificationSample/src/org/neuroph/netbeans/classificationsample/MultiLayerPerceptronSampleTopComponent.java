package org.neuroph.netbeans.classificationsample;

import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.Transferable;
import java.awt.datatransfer.UnsupportedFlavorException;
import java.awt.dnd.DnDConstants;
import java.awt.dnd.DropTarget;
import java.awt.dnd.DropTargetDragEvent;
import java.awt.dnd.DropTargetDropEvent;
import java.awt.dnd.DropTargetEvent;
import java.awt.dnd.DropTargetListener;
import java.io.IOException;
import java.util.logging.Logger;
import org.netbeans.api.settings.ConvertAsProperties;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.data.DataSet;
import org.neuroph.netbeans.visual.TrainingController;
import org.neuroph.netbeans.visual.NeuralNetAndDataSet;
import org.neuroph.netbeans.project.NeurophProjectFilesFactory;
import org.neuroph.nnet.learning.LMS;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.openide.loaders.DataObject;
import org.openide.util.Exceptions;
import org.openide.util.Lookup;
import org.openide.util.NbBundle;
import org.openide.util.lookup.AbstractLookup;
import org.openide.util.lookup.InstanceContent;
import org.openide.util.lookup.ProxyLookup;
import org.openide.windows.TopComponent;
import org.openide.windows.WindowManager;

/**
 * Top component which displays something.
 */
@ConvertAsProperties(dtd = "-//org.neuroph.netbeans.classificationsample.mlperceptron//MultiLayerPerceptronSample//EN",
autostore = false)
public final class MultiLayerPerceptronSampleTopComponent extends TopComponent implements LearningEventListener {

    private static MultiLayerPerceptronSampleTopComponent instance;
    /** path to the icon used by the component and its open action */
//    static final String ICON_PATH = "SET/PATH/TO/ICON/HERE";
    private static final String PREFERRED_ID = "MultiLayerPerceptronSampleTopComponent";

    
    
    public MultiLayerPerceptronSampleTopComponent() {
        initComponents();
        setName(NbBundle.getMessage(MultiLayerPerceptronSampleTopComponent.class, "CTL_MultiLayerPerceptronSampleTopComponent"));
        setToolTipText(NbBundle.getMessage(MultiLayerPerceptronSampleTopComponent.class, "HINT_MultiLayerPerceptronSampleTopComponent"));
//        setIcon(ImageUtilities.loadImage(ICON_PATH, true));
        putClientProperty(TopComponent.PROP_UNDOCKING_DISABLED, Boolean.TRUE);
                putClientProperty(TopComponent.PROP_UNDOCKING_DISABLED, Boolean.TRUE);
        content = new InstanceContent();
        aLookup = new AbstractLookup(content);

    }

    /** This method is called from within the constructor to
     * initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is
     * always regenerated by the Form Editor.
     */
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 770, Short.MAX_VALUE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 600, Short.MAX_VALUE)
        );
    }// </editor-fold>//GEN-END:initComponents

    // Variables declaration - do not modify//GEN-BEGIN:variables
    // End of variables declaration//GEN-END:variables
    /**
     * Gets default instance. Do not use directly: reserved for *.settings files only,
     * i.e. deserialization routines; otherwise you could get a non-deserialized instance.
     * To obtain the singleton instance, use {@link #findInstance}.
     */
    public static synchronized MultiLayerPerceptronSampleTopComponent getDefault() {
        if (instance == null) {
            instance = new MultiLayerPerceptronSampleTopComponent();
        }
        return instance;
    }

    /**
     * Obtain the MultiLayerPerceptronSampleTopComponent instance. Never call {@link #getDefault} directly!
     */
    public static synchronized MultiLayerPerceptronSampleTopComponent findInstance() {
        TopComponent win = WindowManager.getDefault().findTopComponent(PREFERRED_ID);
        if (win == null) {
            Logger.getLogger(MultiLayerPerceptronSampleTopComponent.class.getName()).warning(
                    "Cannot find " + PREFERRED_ID + " component. It will not be located properly in the window system.");
            return getDefault();
        }
        if (win instanceof MultiLayerPerceptronSampleTopComponent) {
            return (MultiLayerPerceptronSampleTopComponent) win;
        }
        Logger.getLogger(MultiLayerPerceptronSampleTopComponent.class.getName()).warning(
                "There seem to be multiple components with the '" + PREFERRED_ID
                + "' ID. That is a potential source of errors and unexpected behavior.");
        return getDefault();
    }

    @Override
    public int getPersistenceType() {
        return TopComponent.PERSISTENCE_ALWAYS;
    }
    
        @Override
    public Lookup getLookup() {
        return new ProxyLookup(new Lookup[]{
            super.getLookup(),
            aLookup
        });
    }
    

    //check for this
    @Override
    public void componentOpened() {

    }

    @Override
    public void componentClosed() {
        // TODO add custom code on component closing
    }

    void writeProperties(java.util.Properties p) {
        // better to version settings since initial version as advocated at
        // http://wiki.apidesign.org/wiki/PropertyFiles
        p.setProperty("version", "1.0");
        // TODO store your settings
    }

    Object readProperties(java.util.Properties p) {
        if (instance == null) {
            instance = this;
        }
        instance.readPropertiesImpl(p);
        return instance;
    }

    private void readPropertiesImpl(java.util.Properties p) {
        String version = p.getProperty("version");
        // TODO read your settings according to their version
    }

    @Override
    protected String preferredID() {
        return PREFERRED_ID;
    }

    private InputSpacePanel inputSpacePanel;
  
    MultiLayerPerceptronClassificationSamplesPanel controllsPanel;
    SettingsTopComponent stc;
    PerceptronSampleTrainingSet pst;
    private DataSet trainingSet;

    public DataSet getTrainingSet() {
        return trainingSet;
    }

    public void setTrainingSet(DataSet trainingSet) {
        this.trainingSet = trainingSet;
    }
    public int tsCount = 0;
    NeuralNetwork neuralNetwork;
    public NeuralNetAndDataSet neuralNetAndDataSet;
    public TrainingController trainingController;

    Thread firstCalculation = null;
    int iterationCounter = 0;
    InstanceContent content;
    AbstractLookup aLookup;
    boolean trainSignal = false;
    public boolean isTrainSignal() {
        return trainSignal;
    }

    public InputSpacePanel getInputSpacePanel() {
        return inputSpacePanel;
    }

    public void setInputSpacePanel(InputSpacePanel inputSpacePanel) {
        this.inputSpacePanel = inputSpacePanel;
    }
    
    public void setTrainSignal(boolean trainSignal) {
        this.trainSignal = trainSignal;
    }
    
    public void initializePanel(boolean positiveCoordinates){
        if(inputSpacePanel!=null){
            this.remove(inputSpacePanel);
        }
        inputSpacePanel = new InputSpacePanel();
        inputSpacePanel.setPositive(positiveCoordinates);
        inputSpacePanel.setSize(570, 570);
        add(inputSpacePanel);
        inputSpacePanel.setLocation(0, 0);
        repaint();
    }


    /** Creates new form BackpropagationSample */
    public void setTrainingSetForMultiLayerPerceptronSample(PerceptronSampleTrainingSet ps) {
        setSize(770, 600);
        trainingSet = new DataSet(2, 1);
        this.pst = ps;   
        stc = SettingsTopComponent.findInstance();
        stc.initializePanel(this);
        controllsPanel = stc.getControllsPanel();
        stc.open();       
        initializePanel(false);

// DragNDrop - start
        this.dtListener = new DTListener();

        this.dropTarget = new DropTarget(
                  this,
                  this.acceptableActions,
                  this.dtListener,
                  true);

        this.dropTarget2 = new DropTarget(
                  inputSpacePanel,
                  this.acceptableActions,
                  this.dtListener,
                  true);

        this.dropTarget3 = new DropTarget(
                  controllsPanel,
                  this.acceptableActions,
                  this.dtListener,
                  true);
// DragNDrop - end
    }


    boolean f = false;


    public void createNeuralNetworkFile(NeuralNetwork neuralNetwork){
        NeurophProjectFilesFactory.getDefault().createNeuralNetworkFile(neuralNetwork);
    }
    
    public void customDataSetCheck() {
        if (inputSpacePanel.isPointDrawed()) {
            trainingSet = inputSpacePanel.getTrain();
            tsCount++;
            trainingSet.setLabel("MlpSampleTrainingSet" + tsCount);

        }
    }

    public void sampleTrainingSetFileCheck() {
        if (inputSpacePanel.isPointDrawed()) {
            NeurophProjectFilesFactory.getDefault().createTrainingSetFile(trainingSet);
            inputSpacePanel.setPointDrawed(false);
        }
    }
    
    public void showPointsOptionCheck(){
         if (MultiLayerPerceptronClassificationSamplesPanel.SHOW_POINTS && inputSpacePanel.isAllPointsRemoved()) {
            try {
                inputSpacePanel.setAllPointsRemoved(false);
                drawPointsFromTrainingSet(trainingSet);
            } catch (Exception e) {
            }
        }
    }

    /*
     * goes to train button
     */
    public void visualizationPreprocessing() {
        controllsPanel.visualizationPreprocessing();
    }
    
    public void setVisualize(boolean flag){
        inputSpacePanel.setVisualize(flag);
    }
    
    public void trainingPreprocessing(){       
        trainSignal = true;

        neuralNetAndDataSet = new NeuralNetAndDataSet(neuralNetwork, trainingSet);
        trainingController = new TrainingController(neuralNetAndDataSet);

        neuralNetwork.getLearningRule().addListener(this);// dodaje se observer na mrezu i pri njenoj promeni poziva se metoda update u ovoj klasi

        trainingController.setLmsParams(controllsPanel.getLearningRate(), controllsPanel.getMaxError(), controllsPanel.getMaxIteration());

        LMS learningRule = (LMS) this.neuralNetAndDataSet.getNetwork().getLearningRule();

        if (learningRule instanceof MomentumBackpropagation) {
            ((MomentumBackpropagation) learningRule).setMomentum(controllsPanel.getMomentum());
        }
        getInputSpacePanel().setNeuronColors(neuralNetwork, true);
    }

    public void stop() // zaustavljanje treniranja pozvano iz source panela
    {
        neuralNetAndDataSet.stopTraining();
    }

    public void clear() //  restartovanje inputSpacePanela pozvano iz source panela
    {       
        inputSpacePanel.clearPoints();
    }

    public void setPointDrawed(boolean drawed) {
        inputSpacePanel.setPointDrawed(drawed);
    }

    // poziva se iz update metode za ulaze od (0,0) pa sve do (1,1) povecavajuci se za 0.02 racuna se izlaz mreze, ovo je vid testiranja na uvek istim podacima
    public void imagePlaying(NeuralNetwork nn) {

        if (nn != null) {
            inputSpacePanel.setNeuralNetwork(nn);
            double xVal;
            double yVal;
            double size;
            double coef;
            if(inputSpacePanel.isPositive()){
                xVal = 0.0;
                yVal = 1.0;
                size = 50;
                coef = 0.02;
            }
            else{
                xVal = -1.0;
                yVal = 1.0;
                size = 57;
                coef = 0.0357142857142857;
            }
           
            for (int i = 0; i < size; i++) // racunanje izlaza mreze za 2500 ulaza
            {
                for (int j = 0; j < size; j++) {
                    double x = xVal + i * coef;        // vrednosti ulaza x1 i x2
                    double y = yVal - j * coef;
                    //neuralNetwork
                    nn.setInput(new double[]{x, y});
                    nn.calculate();
                    int lastLayerIdx = nn.getLayersCount() - 1;
                    double v = nn.getLayerAt(lastLayerIdx).getNeuronAt(0).getOutput();    // izlaz iz mreze za ulaze x1 i x2
                    //     System.out.println("ULAZ ["+i+"]["+j+"] JE:"+"["+x+"]"+"["+y+"] A IZLAZ JE "+v);
                    inputSpacePanel.setGridPoints(i, j, v); // pozivanje metode koja treba da pripremi crtanje u inputSpacePanelu

                }
            }
        }
    }
    

    
    @Override
    public void handleLearningEvent(LearningEvent le) {
        iterationCounter++;
        if (iterationCounter % 10 == 0) {
            NeuralNetwork nnet = neuralNetAndDataSet.getNetwork();

            nnet.pauseLearning();                             // pauza
            imagePlaying(nnet);                                  //  racunanje odgovora mreze i pozivanje crtanja
            nnet.resumeLearning();                            //  nastavak ucenja
        }
    }

    


// DragNDrop - start
    // DnD liseneri
    class DTListener implements DropTargetListener {


        @Override
        public void dragEnter(DropTargetDragEvent dtde) {
            dtde.acceptDrag(dtde.getDropAction());
        }

        @Override
        public void dragExit(DropTargetEvent dte) {
        }

        @Override
        public void dragOver(DropTargetDragEvent dtde) {
            dtde.acceptDrag(dtde.getDropAction());
        }

        public void dropActionChanged(DropTargetDragEvent dtde) {
            dtde.acceptDrag(dtde.getDropAction());
        }

        @Override
        public void drop(DropTargetDropEvent e) {
            Transferable t = e.getTransferable();
            DataFlavor dataSetflavor = t.getTransferDataFlavors()[1];
            try {
                DataObject dataObject = (DataObject) t.getTransferData(dataSetflavor);
                DataSet dataSet = dataObject.getLookup().lookup(DataSet.class);
                NeuralNetwork neuralNet = dataObject.getLookup().lookup(NeuralNetwork.class);               
                if (dataSet != null) {
                    clear();
                    setPointDrawed(false);
                    getInputSpacePanel().setDrawingLocked(true);
                    trainingSet = dataSet;
                    informationCheck(neuralNetwork, trainingSet);
                    boolean positive = true;
                    loop:
                    for (int i = 0; i < trainingSet.size(); i++) {
                        double[] inputs = trainingSet.getRowAt(i).getInput();
                        for (int j = 0; j < inputs.length; j++) {
                            if(inputs[j]<0){
                                positive = false;
                                break loop;
                            }
                            
                        }
                        
                    }
                    initializePanel(positive); 
                    stc.getControllsPanel().setCheckPoints(positive);
                    getInputSpacePanel().drawPointsFromTrainingSet(trainingSet);
                }

                if (neuralNet != null) {
                    neuralNetwork = neuralNet; 
                    informationCheck(neuralNetwork, trainingSet);
                }

                if ((trainingSet != null) && (neuralNetwork != null)) {
                    
                    removeNetworkAndDataSetFromContent();                    
                    trainingPreprocessing();     
                    
                    content.add(neuralNetAndDataSet);
                    content.add(trainingController);
                    informationCheck(neuralNetwork, trainingSet);
                    
                    MultiLayerPerceptronSampleTopComponent.this.requestActive();
                }

            } catch (UnsupportedFlavorException | IOException ex) {
                Exceptions.printStackTrace(ex);
            }

            e.dropComplete(true);
        }
    }
    
    public void informationCheck(NeuralNetwork neuralNetvork, DataSet dataSet){
        
        MultiLayerPerceptronClassificationSamplesPanel mlp =
                stc.getControllsPanel();

        if (dataSet != null) {
            if (dataSet.getLabel()!=null) {
                mlp.setDataSetInformation(dataSet.getLabel());
            }else{
                mlp.setDataSetInformation("Not selected");
            }
        } else {
            mlp.setDataSetInformation("Not selected");
        }

        if (neuralNetvork != null) {
            if (neuralNetvork.getLabel()!=null) {
                mlp.setNeuralNetworkInformation(neuralNetvork.getLabel());
            }else{
                mlp.setNeuralNetworkInformation("Not selected");
            }
        } else {
            mlp.setNeuralNetworkInformation("Not selected");
        }
        
    }
    public void removeNetworkAndDataSetFromContent(){
        try {
            content.remove(neuralNetAndDataSet);
            content.remove(trainingController);
        } catch (Exception ex) {
        }
        MultiLayerPerceptronSampleTopComponent.this.requestActive();
    }
    
    private DropTarget dropTarget;
    private DropTarget dropTarget2;
    private DropTarget dropTarget3;
    private DropTargetListener dtListener;
    private int acceptableActions = DnDConstants.ACTION_COPY;
    
    public boolean isAllPointsRemoved(){
        return inputSpacePanel.isAllPointsRemoved();
    }
    public boolean isPointDrawed(){
        return inputSpacePanel.isPointDrawed();
    }
    public boolean isDrawingLocked(){
        return inputSpacePanel.isDrawingLocked();
    }
    public void setDrawingLocked(boolean flag){
        inputSpacePanel.setDrawingLocked(flag);
    }
     public  void drawPointsFromTrainingSet(DataSet dataSet){

         try{
             inputSpacePanel.setAllPointsRemoved(false);
             inputSpacePanel.drawPointsFromTrainingSet(dataSet);
         }
         catch(Exception e){
             
         }
        
    }
}